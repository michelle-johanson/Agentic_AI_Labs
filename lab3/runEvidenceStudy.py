"""Run Camden's existing checks, optionally recording public LangSmith evidence.

python runEvidenceStudy.py
python runEvidenceStudy.py --trace
python runEvidenceStudy.py --trace --live --limit 6

The controlled comparison measures validation, not LLM accuracy. The live
comparison uses the same model/temperature for both variants and requires
human review of meaning, category, and omissions.
"""
import argparse
import asyncio
import inspect
import json
import os
from pathlib import Path
import time
from uuid import uuid4

from pydantic import Field, ValidationInfo, create_model, field_validator
from langsmith import Client, traceable, tracing_context
from langsmith.run_helpers import get_current_run_tree

import docAnalyzerCommon as common
import evalEvidence
import script2PydanticStructured as structured


OUTPUT_DIR = common.LAB_DIR / 'files'


class LegacyClimateDisclosure(evalEvidence.LegacyClimateDisclosure):
    """Original string schema, including its description-echo cleanup."""

    @field_validator(*common.EXPECTED_KEYS, mode='after')
    @classmethod
    def drop_parroted_description(cls, items: list[str], info: ValidationInfo):
        description = common.ClimateDisclosure.model_fields[info.field_name].description or ''
        return [item for item in items if common._normalize(item) != common._normalize(description)]


# The legacy evaluation schema has no descriptions. Restore the original
# descriptions for the live model comparison so only the evidence requirement
# and source checking change, not category guidance.
LiveLegacyDisclosure = create_model(
    'ClimateDisclosure', __base__=LegacyClimateDisclosure,
    **{key: (list[str], Field(default_factory=list,
                            description=common.ClimateDisclosure.model_fields[key].description))
       for key in common.EXPECTED_KEYS},
)


def controlled_quote(chunk: str, case: str) -> str:
    quote = ' '.join(chunk.split())[:160].rsplit(' ', 1)[0]
    variants = {
        'verbatim': quote,
        'whitespace_only': '\n\t'.join(quote.split()),
        'invented_quote': 'The company emitted 987654321 tonnes of CO2e in 2099.',
        'altered_punctuation': quote + ' !!!',
        'altered_case': quote.swapcase(),
        'stitched_passages': quote[:40] + ' ... ' + quote[-40:],
    }
    return variants[case]


@traceable(name='controlled_quote_validation')
def check_quote(file: str, chunk_index: int, source: str, case: str, quote: str, expected: bool):
    legacy = evalEvidence.LegacyClimateDisclosure.model_validate({'climate_risks':[quote]})
    before = bool(legacy.climate_risks)
    disclosure = evalEvidence.make_disclosure(quote)
    after = evalEvidence.accepts(disclosure, source)
    assert after == expected
    run = get_current_run_tree()
    return {'test_type':'controlled validation; no LLM inference', 'case':case,
            'before_accepted':before,'after_accepted':after,'expected':expected,
            'passed':after == expected,'run_id':str(run.id) if run else None}


@traceable(name='controlled_quote_summary')
def controlled_summary(rows):
    invalid = [row for row in rows if not row['expected']]
    valid = [row for row in rows if row['expected']]
    run = get_current_run_tree()
    return {'test_type':'controlled validation; not model accuracy',
            'invalid_tested':len(invalid),
            'before_invalid_accepted':sum(row['before_accepted'] for row in invalid),
            'after_invalid_accepted':sum(row['after_accepted'] for row in invalid),
            'valid_tested':len(valid),
            'before_valid_accepted':sum(row['before_accepted'] for row in valid),
            'after_valid_accepted':sum(row['after_accepted'] for row in valid),
            'run_id':str(run.id) if run else None}


def run_controlled():
    # Re-run all earlier tests, including retry bounds and scripts 3/4 mocks.
    evalEvidence.run()
    report = json.loads((OUTPUT_DIR/'evidence_eval_results.json').read_text())
    texts = {file:common.chunk_text(common.extract_text_from_pdf(file)) for file in common.list_pdfs()}
    rows=[]
    for row in report['rows']:
        chunk = texts[row['file']][row['chunk_index']]
        result=check_quote(row['file'],row['chunk_index'],chunk,row['case'],
                           controlled_quote(chunk,row['case']),row['expected'])
        rows.append({**row,**result})
    summary=controlled_summary(rows)
    return {'summary':summary,'rows':rows},texts


@traceable(name='live_extraction_comparison', process_inputs=lambda inputs: {
    key:value for key,value in inputs.items()
    if key not in {'baseline_chain','evidence_chain'}
})
def extract_live(source, file, chunk_index, variant, model_name, baseline_chain, evidence_chain):
    # Record source text and settings; exclude the Python chain handles.
    run=get_current_run_tree()
    if run:
        run.inputs = {'source':source,'file':file,'chunk_index':chunk_index,
                      'variant':variant,'model':model_name,'temperature':0.3}
    started=time.perf_counter()
    disclosure=None
    error=None
    failed_attempts=[]
    for attempt in range(1,structured.MAX_RETRIES+1):
        try:
            if variant == 'before':
                disclosure=baseline_chain.invoke({'text':source},config={
                    'run_name':f'{file}_chunk{chunk_index}_{variant}_attempt{attempt}',
                    'tags':['evidence-comparison',variant]})
            else:
                candidate=evidence_chain.invoke({'text':source,'retry_feedback':(
                    'The previous response failed validation. Return the required objects '
                    'and use only exact quotes from the source text; omit unsupported findings.'
                    if error else '')},config={
                        'run_name':f'{file}_chunk{chunk_index}_{variant}_attempt{attempt}',
                        'tags':['evidence-comparison',variant]})
                disclosure=common.validate_evidence(candidate,source)
            error=None
            break
        except Exception as exc:
            disclosure=None
            error=str(exc)
            failed_attempts.append({'attempt':attempt,'error':error})
    output=disclosure.model_dump() if disclosure is not None else None
    count=sum(len(value) for value in output.values()) if output else 0
    return {'file':file,'chunk_index':chunk_index,'variant':variant,
            'status':'ok' if disclosure is not None else 'failed','attempts':attempt,
            'seconds':round(time.perf_counter()-started,3),'output':output,
            'error':error,'failed_attempts':failed_attempts,'finding_count':count,
            'source_matched_quotes':count if variant == 'after' and output else 0,
            'run_id':str(run.id) if run else None,
            'human_review':{'unsupported_findings':None,'missed_disclosures':None,
                            'incorrect_categories':None,'notes':''}}


def run_live(texts, model, limit):
    from langchain_ollama import ChatOllama
    from langchain_core.prompts import PromptTemplate
    # Match Camden's 0.3 temperature and three-attempt budget on both sides.
    llm=ChatOllama(model=model,temperature=0.3)
    baseline=PromptTemplate.from_template(common.EXTRACTION_INSTRUCTIONS) | llm.with_structured_output(LiveLegacyDisclosure)
    evidence=structured.PROMPT | llm.with_structured_output(common.ClimateDisclosure)
    # One chunk per company before taking each company's next eligible chunk.
    candidates={file:[(i,c) for i,c in enumerate(chunks) if common.keyword_hit(c)]
                for file,chunks in texts.items()}
    samples=[]
    depth=0
    while len(samples)<limit:
        added=False
        for file,items in candidates.items():
            if depth<len(items) and len(samples)<limit:
                index,chunk=items[depth]
                samples.append((file,index,chunk))
                added=True
        if not added:
            break
        depth+=1
    if not samples:
        raise ValueError('No keyword-selected passages found')
    rows=[]
    stamp=time.strftime('%Y%m%d_%H%M%S')+'_'+uuid4().hex[:6]
    path=OUTPUT_DIR/f'live_evidence_{stamp}.json'
    for number,(file,index,source) in enumerate(samples,1):
        # Alternate order to reduce a consistent model-loading/order advantage.
        order=('before','after') if number%2 else ('after','before')
        for variant in order:
            print(f'[{number}/{len(samples)}] {file} chunk {index}: {variant}',flush=True)
            row=extract_live(source,file,index,variant,model,baseline,evidence)
            rows.append({**row,'source':source})
            path.write_text(json.dumps({'model':model,'temperature':0.3,
                'note':'Small convenience sample. Human review needed; source matching is not semantic accuracy.',
                'rows':rows},indent=2),encoding='utf-8')
            print(f"  {row['status']}; {row['finding_count']} findings; {row['attempts']} attempts; {row['seconds']} seconds",flush=True)
    print('\nVariant   Passages  Failed  Findings  Attempts  Total seconds')
    for variant in ('before','after'):
        subset=[row for row in rows if row['variant']==variant]
        print(f"{variant:<9} {len(subset):>8} {sum(r['status']=='failed' for r in subset):>7} "
              f"{sum(r['finding_count'] for r in subset):>9} {sum(r['attempts'] for r in subset):>9} "
              f"{sum(r['seconds'] for r in subset):>14.3f}")
    print('Saved live inputs/outputs for human review:',path,flush=True)
    return rows,path


async def _await_result(result):
    return await result if inspect.isawaitable(result) else result


async def _resolve_link(client, run_id, project_id, share):
    """Use project coordinates and the installed SDK's supported v2 resource."""
    resource = client.runs
    # Current docs call this retrieve(); the installed SDK calls it retrieve_v2().
    retrieve = getattr(resource, 'retrieve', None) or resource.retrieve_v2
    run = await _await_result(retrieve(
        str(run_id), project_id=project_id,
        selects=['ID','TRACE_ID','START_TIME','SHARE_URL'],
    ))
    if not share:
        result = await _await_result(resource.get_url(
            str(run_id), project_id=project_id, trace_id=str(run.trace_id or run_id),
        ))
        return 'private_url', result.url
    if run.share_url:
        return 'public_url', run.share_url
    result = await _await_result(resource.share.create(
        str(run_id), session_id=project_id, trace_id=str(run.trace_id or run_id),
    ))
    if not result.share_token:
        raise RuntimeError('LangSmith did not return a share token')
    # Ask the service for its application URL instead of constructing one.
    updated = await _await_result(retrieve(
        str(run_id), project_id=project_id, selects=['SHARE_URL'],
    ))
    if not updated.share_url:
        raise RuntimeError('Public link is not visible yet')
    return 'public_url', updated.share_url


async def _link_with_retry(client, run_id, project_id, share):
    for attempt in range(5):
        try:
            return await _resolve_link(client, run_id, project_id, share)
        except Exception as exc:
            text = str(exc).lower()
            transient = any(marker in text for marker in (
                '404','not found','not visible yet','429','timeout','503','502','504',
            ))
            if not transient or attempt == 4:
                raise
            await asyncio.sleep(2 ** attempt)


async def _collect_links(client, rows, share, project_id, existing=None):
    links=[]
    saved = {row['run_id']:row for row in (existing or [])}
    for row in rows:
        run_id=row.get('run_id')
        if not run_id:
            continue
        entry=dict(saved.get(run_id, {})) or {'case':row.get('case','live extraction'),
               'file':row.get('file'),'variant':row.get('variant'),'run_id':run_id}
        if share and entry.get('public_url'):
            links.append(entry)
            continue
        try:
            key, url = await _link_with_retry(client,run_id,project_id,share)
            entry[key] = url
            entry.pop('link_error',None)
        except Exception as exc:
            entry['link_error']=str(exc)
        links.append(entry)
    return links


def save_links(client, rows, share, project_name, existing=None):
    client.flush(timeout=30)
    project_id = str(client.read_project(project_name=project_name).id)
    # Keep the SDK's async HTTP client in one event loop for the entire batch.
    return asyncio.run(_collect_links(client,rows,share,project_id,existing))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace',action='store_true',help='Send public filing excerpts and results to LangSmith')
    parser.add_argument('--share',action='store_true',help='Create PUBLIC links for recorded runs')
    parser.add_argument('--live',action='store_true',help='Also run a live before/after model comparison')
    parser.add_argument('--repair-links',type=Path,
                        help='Repair a saved link file without rerunning tests or models')
    parser.add_argument('--model',default=structured.MODEL_NAME)
    parser.add_argument('--limit',type=int,default=6,help='Live passage pairs (up to 6 model calls per pair)')
    args=parser.parse_args()
    if args.limit<1 or args.limit>20:
        parser.error('--limit must be between 1 and 20')
    if args.share and not args.trace:
        parser.error('--share requires --trace')
    if args.repair_links and not args.trace:
        parser.error('--repair-links requires --trace')
    if args.trace and not os.environ.get('LANGSMITH_API_KEY'):
        parser.error('Set LANGSMITH_API_KEY in this terminal before --trace')
    OUTPUT_DIR.mkdir(exist_ok=True)
    client=Client() if args.trace else None
    if client:
        # Verify credentials before spending time on PDFs or model calls.
        client.info
        list(client.list_projects(limit=1))
    project=os.environ.get('LANGSMITH_PROJECT','lab3-camden-evidence')
    if args.repair_links:
        path = args.repair_links.resolve()
        if path.parent != OUTPUT_DIR.resolve():
            parser.error('--repair-links must name a JSON file in lab3/files')
        payload = json.loads(path.read_text(encoding='utf-8'))
        project = payload['project']
        payload['links'] = save_links(client,payload['links'],args.share,project,
                                     existing=payload['links'])
        path.write_text(json.dumps(payload,indent=2),encoding='utf-8')
        for entry in payload['links']:
            if entry.get('case') == 'controlled summary':
                print('Summary link:',entry.get('public_url',entry.get('private_url',entry.get('link_error'))))
        failed = sum('link_error' in entry for entry in payload['links'])
        print(f'Repaired link file: {len(payload["links"])-failed}/{len(payload["links"])} links available')
        print('Saved:',path)
        return
    with tracing_context(enabled=args.trace,client=client,project_name=project):
        controlled,texts=run_controlled()
        summary=controlled['summary']
        print('\nControlled validation: invalid quotes accepted',
              f"{summary['before_invalid_accepted']}/{summary['invalid_tested']} before ->",
              f"{summary['after_invalid_accepted']}/{summary['invalid_tested']} after")
        stamp=time.strftime('%Y%m%d_%H%M%S')+'_'+uuid4().hex[:6]
        path=OUTPUT_DIR/f'controlled_evidence_{stamp}.json'
        path.write_text(json.dumps(controlled,indent=2),encoding='utf-8')
        live,path_live=run_live(texts,args.model,args.limit) if args.live else ([],None)
    if client:
        summaries=[{'case':'controlled summary',**summary}]
        links=save_links(client,summaries+controlled['rows']+live,args.share,project)
        link_path=OUTPUT_DIR/f'evidence_links_{stamp}.json'
        link_path.write_text(json.dumps({'project':project,'links':links},indent=2),encoding='utf-8')
        print('Saved LangSmith links:',link_path)
        if links:
            print('Summary link:',links[0].get('public_url',links[0].get('private_url',links[0].get('link_error'))))
        if any('link_error' in link for link in links):
            print('Some links failed. Runs may still exist; check the saved link errors and LangSmith project.')
    print('Controlled results:',path)
    print('Live accuracy must be judged against the saved sources; no gain is assumed.')


if __name__=='__main__':
    main()

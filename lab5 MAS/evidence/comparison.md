# Lab 5 live comparison

Six completed runs on 5 November 2026. Same CASE, `qwen2.5:7b`, and two-round limit. Token counts are sums of `usage_metadata` on model responses. No dollar cost was reported.

An aborted `merged-smoke` and a detached `merged-01` that died before the first argument finished are not included. The counted `merged-01` is the completed run.

| Run | Seconds | History entries | Searches | Input tokens | Output tokens | Recommendation in the first sentence | Public trace |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-01 | 126.9 | none stored | 3 | 16959 | 1336 | No. First sentence names the cost/flexibility tradeoff. Later text says BUY. | https://smith.langchain.com/public/8d26bcc4-0752-4679-83ac-fc07f9e0287d/r |
| baseline-02 | 144.2 | none stored | 3 | 17114 | 1588 | No. Tradeoff first. Later text says BUILD. | https://smith.langchain.com/public/bd9f8567-685a-4435-ae69-ebf4f8a05669/r |
| baseline-03 | 127.9 | none stored | 3 | 17239 | 1352 | No. Tradeoff first. Later text says BUILD. | https://smith.langchain.com/public/a984d17b-1dda-492d-a2ef-ce7bcf957b8a/r |
| merged-01 | 286.5 | 6 | 4 | 29847 | 3560 | No. Tradeoff heading first. Later text says BUILD. | https://smith.langchain.com/public/afac2519-ade3-4e2b-9fc6-3086508f5d2b/r |
| merged-02 | 339.1 | 6 | 5 | 31079 | 4199 | No. Tradeoff heading first. Later text says BUY for a 15-person team that is not in the CASE. | https://smith.langchain.com/public/a44bfa64-6620-4290-b960-0a4d8f3e4f9c/r |
| merged-03 | 351.0 | 6 | 2 | 19815 | 5461 | No. Opens with a BUILD summary. BUILD appears later. | https://smith.langchain.com/public/f0ae147e-8b30-473d-93ee-c51ad9700d7d/r |

Mean time: baseline 133.0 s, merged 325.5 s.

## What the merged traces showed

- Every merged moderator input contained Round 1 BUILD, BUY, FACT CHECK and Round 2 BUILD, BUY, FACT CHECK. The baseline-01 moderator input had the two final arguments and no round labels.
- Both second-round advocates in all three merged runs received the first fact-check report, including its program-written search record.
- Checkers searched broad cost comparisons (`cost comparison of building custom CRM vs SaaS CRM`, `cost of custom CRM in-house vs SaaS CRM`, `cost of ownership SaaS CRM vs in-house CRM`). They did not isolate one external claim and leave the CASE dollar assumptions alone.
- merged-01's returned snippet from bitvea.com discusses a $100,000 customization cost versus a $24,000 first-year SaaS cost. The full pages were not opened. The checker's writeup then added other figures, including $20,000–$80,000 and a Jiss Tech comparison. Snippet support was not established for every number in the report.
- merged-01 and merged-03 second checkers made no tool call. The program appended the unverified warning. merged-01's second report still said two claims were supported and pointed the ActiveCampaign review claim at `https://bitvea.com/en/blog/crm-custom-vs-saas`, which is not a review page.
- No merged run contained a clear contradicted verdict that a later advocate withdrew. Correction opportunity: not a clean one. Later arguments repeated or absorbed the checker's web summary. In merged-03 the BUY advocate argued for building.

Public pages were requested without an Authorization header and returned HTTP 200. The HTML shells did not contain `tvly-` or `lsv2_`. The merged-01 trace was also opened in a browser while logged out of the workspace view; the run name and fact-check text rendered.

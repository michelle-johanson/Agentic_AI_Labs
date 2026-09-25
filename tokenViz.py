from transformers import AutoTokenizer
from colorama import init, Fore, Style

init()

models = [
    'bert-base-uncased',
    'GPT2',
    'openai-gpt',
    'roberta-base',
    'xlnet-base-cased',
    'albert-base-v2',

]

FORES = [ 
    Fore.BLACK, 
    Fore.RED, 
    Fore.GREEN, 
    Fore.YELLOW, 
    Fore.BLUE, 
    Fore.MAGENTA, 
    Fore.CYAN, 
    Fore.WHITE 
]

text = """To be or not to be, that BE the Question!!!!  I love Cosmo23765%&%^&%^"""

for model in models:
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokens = tokenizer.tokenize(text)
    print(f"Tokenization using {model}:")

    masterTokens = ""
    
    for i, token in enumerate(tokens):
        color = FORES[i%len(FORES)]
        masterTokens += color + token
    
    masterTokens += Style.RESET_ALL
    print(masterTokens)  
    print("\n")
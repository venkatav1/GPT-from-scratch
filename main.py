### MAIN FUNCTION
import argparse, configparser
import torch
import tiktoken
from data_prep import data_prep, text_to_token_ids, token_ids_to_text
from model import GPTModel
from train import train_model_simple
from inference import generate

torch.manual_seed(123)


def train(config):   

    config_dict = dict(config.items('TRAINING'))
    config_dict['vocab_size'] = int(config_dict['vocab_size'])
    config_dict['context_length'] = int(config_dict['context_length'])
    config_dict['emb_dim'] = int(config_dict['emb_dim'])
    config_dict['n_heads'] = int(config_dict['n_heads'])
    config_dict['n_layers'] = int(config_dict['n_layers'])
    config_dict['drop_rate'] = float(config_dict['drop_rate'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(123) # For reproducibility due to the shuffling in the data loader

    train_loader, val_loader = data_prep(config_dict['file_path'],config_dict)
    model = GPTModel(config_dict)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

    num_epochs = 10
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5, model_path=config_dict['model_path']
    )


def sample(config, text):
    config_dict = dict(config.items('INFERENCE'))
    config_dict['vocab_size'] = int(config_dict['vocab_size'])
    config_dict['context_length'] = int(config_dict['context_length'])
    config_dict['emb_dim'] = int(config_dict['emb_dim'])
    config_dict['n_heads'] = int(config_dict['n_heads'])
    config_dict['n_layers'] = int(config_dict['n_layers'])
    config_dict['drop_rate'] = float(config_dict['drop_rate'])
    config_dict['max_new_tokens'] = int(config_dict['max_new_tokens'])
    config_dict['top_k'] = int(config_dict['top_k'])
    config_dict['temperature'] = float(config_dict['temperature'])
    tokenizer=tiktoken.get_encoding("gpt2")
    torch.manual_seed(123)
    model = GPTModel(config_dict)
    model.load_state_dict(torch.load(config_dict['model_path']))
    model.eval();
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(text, tokenizer),
        max_new_tokens=config_dict['max_new_tokens'],
        context_size=config_dict['context_length'],
        top_k=config_dict['top_k'],
        temperature=config_dict['temperature']
    )

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode', '-m',
        help='Set mode: train or sample from trained model',
        default='train',
        type=str,
        choices=['train', 'sample'],
    )
    parser.add_argument(
        '--config', '-c',
        help='Set the full path to the config file.',
        type=str,
    )

    parser.add_argument(
        '--text', '-t',
        help='Prompt the GPT model',
        type=str,
    )

    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    if args.mode == "train":
        train(config)
    if args.mode == "sample":
        sample(config, args.text)
import argparse

parser = argparse.ArgumentParser(description="Prepared Adapter")

parser.add_argument('--peft_method', type=str, choices={'lora','qlora'}, default='qlora')

args = parser.parse_args()

if args.peft_method == 'qlora':
    print(args.peft_method)

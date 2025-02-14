#!/usr/bin/env python3
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/Transformer-XL/XLNet)
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
from tqdm import trange

import torch
import torch.nn.functional as F
import numpy as np

from pytorch_transformers import GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig

from pytorch_transformers import GPT2LMHeadModel, GPT2Tokenizer
from pytorch_transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
from pytorch_transformers import XLNetLMHeadModel, XLNetTokenizer
from pytorch_transformers import TransfoXLLMHeadModel, TransfoXLTokenizer

import random


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig)), ())

MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'xlnet': (XLNetLMHeadModel, XLNetTokenizer),
    'transfo-xl': (TransfoXLLMHeadModel, TransfoXLTokenizer),
}

MODEL_TYPE      = ''
MODEL_NAME_PATH = ''
MODEL           = None

# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PADDING_TEXT = """ In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def set_seed2(seed, n_gpu):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)
        
        

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(model, length, context, num_samples=1, temperature=1, top_k=0, top_p=0.0, is_xlnet=False, device='cpu'):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    with torch.no_grad():
        for _ in trange(length):

            inputs = {'input_ids': generated}
            if is_xlnet: 
                # XLNet is a direct (predict same token, not next token) and bi-directional model by default
                # => need one additional dummy token in the input (will be masked), attention mask and target mapping (see model docstring)
                input_ids = torch.cat((generated, torch.zeros((1, 1), dtype=torch.long, device=device)), dim=1)
                perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float, device=device)
                perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token
                target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float, device=device)
                target_mapping[0, 0, -1] = 1.0  # predict last token
                inputs = {'input_ids': input_ids, 'perm_mask': perm_mask, 'target_mapping': target_mapping}

            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            next_token_logits = outputs[0][0, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
    return generated


def bot_writer(prompt, padding_text=None, length = 30, ):
    #model_type = 'gpt2'
    #model_name_or_path = 'gpt2'
    # =============================================
    model_type = 'xlnet'
    model_name_or_path = 'xlnet-base-cased'
    # =============================================
    #model_type = 'transfo-xl'
    #model_name_or_path = 'transfo-xl-wt103'
    #length = 300
    
    # Controls degree of surprise in final output
    '''
    Float value controlling randomness in boltzmann distribution. Lower 
    temperature results in less random completions. As the temperature 
    approaches zero, the model will become deterministic and repetitive. 
    Higher temperature results in more random completions.
    '''    
    temperature = 1.0
    '''
    Integer value controlling diversity. 1 means only 1 word is considered for 
    each step (token), resulting in deterministic completions, while 40 means 
    40 words are considered at each step. 0 is a special setting meaning no 
    restrictions. 40 generally is a good value.
    '''
    top_k = 0
    top_p = 0.9
    no_cuda = True
    #seed = 42
    seed = random.randint(0,100)
    
    #args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    #set_seed(args)
    set_seed2(seed, n_gpu)
    
    
    model_type = model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[model_type]
    tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
    global MODEL_TYPE   
    global MODEL_NAME_PATH
    global MODEL 

    if MODEL_TYPE !=  model_type or MODEL_NAME_PATH !=  model_name_or_path:
        MODEL_TYPE = model_type
        MODEL_NAME_PATH = model_name_or_path
        
        MODEL = model_class.from_pretrained(MODEL_NAME_PATH)
        MODEL.to(device)
        MODEL.eval()
    
    
    if length < 0:
        length = MAX_LENGTH  # avoid infinite loop

    #while True:
    raw_text = prompt
    if MODEL_TYPE in ["transfo-xl", "xlnet"]:
        # Models with memory likes to have a long prompt for short inputs.
        raw_text = padding_text + raw_text
        
    context_tokens = tokenizer.encode(raw_text)
    out = sample_sequence(
        model=MODEL,
        context=context_tokens,
        length=length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        device=device,
        is_xlnet=bool(MODEL_TYPE == "xlnet"),
    )
    out = out[0, len(context_tokens):].tolist()
    text = tokenizer.decode(out, clean_up_tokenization_spaces=True)
    print(text)
    
    #if prompt:
    #    break
    
    return text


# test
padding_text_1 = """
I think that it depends on the student. Some students have more developed 
brains and it tells them when its time to learn and when its time to have fun. 
The younger the children the more they learn while playing video games and then 
the older they get the less they learn. But also some games are ment to learn 
and develop some math skills. Also some games like Call of Duty helps the 
gamers by spotting the rival and in thr classroom it might help them spot 
something. At my school there is some gamers that look and are smart and then 
there is the gamers that are always alone looking weird and always sleeping 
and they are failing. Out in the world there is the gamers that never go 
outside and do no physical activities. And i'm pretty sure there are them 
that play sports and play video games like NBA2K, MADDEN, FIFA. Those are the 
games I play most of the time. I just now recently started playing Ghost Recon 
and its more of an adventure, and action game and you got to think pretty hard 
about how you are going to beat that mission and you will eventually achieve 
the mission. So I am going for that video games will help you in thr classroom.
"""


padding_text_2 = """
Introduction
First, let's establish there are two approaches to 'AI-based' natural language 
understanding. Essentially, AI-based means we want a computer to make judgements 
about text for us, whether it's essay scoring or translating between languages or 
gauging whether a movie review is positive or negative based on the words.

The first approach is 'traditional, feature-based' AI models. These are classic: a 
human comes up with a list of features (such as counting complex words, counting 
the ratio of complex words to simple words, extracting how often adjectives are 
used, etc.), and the text is summarized by those features and rules. This approach 
is fast, simple to understand, and can yield useful results. The problem, however, 
is that these approaches aren't really using the basic fundamentals of language, 
such as word meaning or even the order of words. 

"""


#prompt = 'video games can help kids to learn.'
prompt = 'Deep learning can help do text generation.'

length = 10
bot_writer(prompt, padding_text_2, length)



def main2():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--padding_text", type=str, default="")
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    set_seed(args)

    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    model.to(args.device)
    model.eval()

    if args.length < 0 and model.config.max_position_embeddings > 0:
        args.length = model.config.max_position_embeddings
    elif 0 < model.config.max_position_embeddings < args.length:
        args.length = model.config.max_position_embeddings  # No generation bigger than model size 
    elif args.length < 0:
        args.length = MAX_LENGTH  # avoid infinite loop

    print(args)
    while True:
        raw_text = args.prompt if args.prompt else input("Model prompt >>> ")
        if args.model_type in ["transfo-xl", "xlnet"]:
            # Models with memory likes to have a long prompt for short inputs.
            raw_text = (args.padding_text if args.padding_text else PADDING_TEXT) + raw_text
        context_tokens = tokenizer.encode(raw_text)
        out = sample_sequence(
            model=model,
            context=context_tokens,
            length=args.length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=args.device,
            is_xlnet=bool(args.model_type == "xlnet"),
        )
        out = out[0, len(context_tokens):].tolist()
        text = tokenizer.decode(out, clean_up_tokenization_spaces=True)
        print(text)
        if args.prompt:
            break
    return text


#if __name__ == '__main__':
#    main()

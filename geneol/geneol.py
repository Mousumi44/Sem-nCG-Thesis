import numpy as np
from typing import Dict, List, Union, cast

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
import json
import os
from peft import PeftModel, PeftConfig
from accelerate.utils import gather_object
import time


from . import prompts_utils
import importlib
importlib.reload(prompts_utils)
from .prompts_utils import *


import openai
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import ast
import json_repair
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import time
import numpy as np
from scipy.spatial import ConvexHull
import re
from collections import defaultdict
# from .openai_keys import *



class LLM:

    def __init__(self, accelerator=None, args=None):
        self.args = args
        self.accelerator = accelerator
        self.model=None
        self.tokenizer=None
        self.ltokenizer=None
        
    # generate positives/negatives
    def generate(self, instructions_inputs_batch):
        if 'gpt' in self.args.gen_model_name_or_path:
            outputs=[]
            for messages in instructions_inputs_batch:
                retry_count = 0
                is_ok = False
                while not is_ok:
                    retry_count += 1
                    try:
                        response = openai.ChatCompletion.create(
                            model=self.args.gen_model_name_or_path,
                            messages=messages,
                            n=self.args.num_gens,
                            # temperature=args.temperature,
                            max_tokens=4096,
                            # stop=stop,
                            # top_p=args.top_p,
                        )
                        is_ok = True
                    except Exception as error:
                        time.sleep(1)
                        if retry_count % 5 ==0:
                            logger.warning(f"OpenAI API retry for {retry_count} times ({error})")
                            continue


                outputs.extend([each_gen["message"]["content"] for each_gen in response["choices"]])
            return outputs
        else:
            if(len(instructions_inputs_batch)>=4):
                # import ipdb; ipdb.set_trace()
                # Assuming 'instructions_inputs_batch' is a list of inputs

                batch_size = 4
                final_outputs = []

                # Split the instructions_inputs_batch into chunks of size 8
                for i in range(0, len(instructions_inputs_batch), batch_size):
                    batch = instructions_inputs_batch[i:i + batch_size]

                    # Tokenize and send to model
                    inputs = self.ltokenizer.apply_chat_template(
                        batch, padding=True, truncation=True, 
                        max_length=3*self.args.max_length, return_tensors="pt", return_dict=True
                    ).to("cuda")
                    
                    # Generate output
                    output = self.model.generate(
                        **inputs, do_sample=True, num_return_sequences=self.args.num_gens, 
                        temperature=self.gtemp, repetition_penalty=1.0, 
                        top_p=self.top_p, max_length=3*self.args.max_length, 
                        return_dict_in_generate=True
                    )
                    
                    # Process output
                    new_output_ids = output.sequences[:, inputs.input_ids.shape[1]:]
                    del inputs; del output

                    outputs = self.tokenizer.batch_decode(
                        new_output_ids, skip_special_tokens=True, 
                        spaces_between_special_tokens=False
                    )
                    
                    del new_output_ids

                    # Clean outputs
                    outputs = [
                        o.replace("assistant\n\n", "", 1) if o.startswith("assistant\n\n") else o 
                        for o in outputs
                    ]
                    
                    # Append the processed outputs to the final list
                    final_outputs.extend(outputs)

                # Now final_outputs will contain all the processed outputs
                outputs=final_outputs

            else:    
                inputs = self.ltokenizer.apply_chat_template(instructions_inputs_batch, padding=True, truncation=True, max_length=3*self.args.max_length, return_tensors="pt", return_dict=True).to("cuda")
                output = self.model.generate(**inputs, do_sample=True, num_return_sequences=self.args.num_gens, temperature=self.gtemp, repetition_penalty=1.0, top_p=self.top_p, max_length=3*self.args.max_length, return_dict_in_generate=True)

                new_output_ids = output.sequences[:, inputs.input_ids.shape[1]:]
                del inputs; del output
                outputs = self.tokenizer.batch_decode(new_output_ids, skip_special_tokens=True, spaces_between_special_tokens=False)
                

                del new_output_ids
                outputs = [o.replace("assistant\n\n", "", 1) if o.startswith("assistant\n\n") else o for o in outputs]
            
        return outputs

    # embedd using generations
    def embed(self, new_sentences_batch):


        inputs = self.tokenizer(new_sentences_batch, padding=True, truncation=True, return_tensors='pt',
                                max_length=4*self.args.max_length, add_special_tokens=False).to("cuda")

        all_embeddings = []
        batch_size = 5
        # Get the total number of sentences
        total_sentences = inputs['input_ids'].shape[0]

        # Iterate through the tokenized inputs in chunks of batch_size
        for i in range(0, total_sentences, batch_size):
            # Create a batch of tokenized inputs
            batch_inputs = {key: tensor[i:i+batch_size] for key, tensor in inputs.items()}
            
            # Get the embeddings based on whether the penultimate layer is used
            if self.args.penultimate_layer!=-1:
                hidden_states = (getattr(self.model, self.embedding_attr) if self.embedding_attr else self.model)(
                    **batch_inputs, output_hidden_states=True, return_dict=True).hidden_states
                last_hidden_state = hidden_states[self.args.penultimate_layer]
            else:
                outputs = (getattr(self.model, self.embedding_attr) if self.embedding_attr else self.model)(**batch_inputs)
                last_hidden_state = outputs[0]
                del outputs
            
            # Append the batch embeddings to the list
            all_embeddings.append(last_hidden_state.detach().cpu())
        
        # Concatenate all the batch embeddings into one tensor
        final_embeddings = torch.cat(all_embeddings, dim=0)


        return final_embeddings, inputs

    def switchon_gen_model(self):
        
        if self.args.method=='b5':
            return

        if 'gpt' in self.args.gen_model_name_or_path:
            import openai 
            

            OPENAI_API_BASE = "None"
            openai.api_key = OPENAI_API_KEY
            openai.organization = OPENAI_ORG_ID

            self.openaitokenizer = AutoTokenizer.from_pretrained("gpt2", fast_tokenizer=False) # TODO: For ChatGPT we should use a different one
            self.model=None
        elif self.args.gen_model_name_or_path is not None:
            if self.model is not None: del self.model
            self.model = AutoModelForCausalLM.from_pretrained(self.args.gen_model_name_or_path, trust_remote_code=True, device_map={"": self.accelerator.process_index}, torch_dtype=self.args.torch_dtype)

            if('mistral' in self.args.gen_model_name_or_path.lower()):
                self.gtemp=0.7
                self.top_p=1
            elif('llama' in self.args.gen_model_name_or_path.lower()):
                self.gtemp=0.6
                self.top_p=0.9
            else:
                assert False, "Model not accepted"

            self.embedding_attr = 'model'
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            print(f"Created {self.args.gen_model_name_or_path}: {self.model.dtype} dtype")


            self.ltokenizer = AutoTokenizer.from_pretrained(self.args.gen_model_name_or_path, padding_side='left')
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.gen_model_name_or_path, padding_side='right')
            if not(self.tokenizer.pad_token) and self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.ltokenizer.pad_token = self.ltokenizer.eos_token

            print('Set pad token to eos token: ' + self.tokenizer.pad_token)
            self.model.eval()
        elif self.args.gen_model_name_or_path is None or self.args.gen_model_name_or_path == '-':
            pass
        else:
            pass

    def switchon_emb_model(self):       

        if self.model is not None: del self.model
        
        self.model = AutoModelForCausalLM.from_pretrained(self.args.model_name_or_path, trust_remote_code=True, device_map={"": self.accelerator.process_index}, torch_dtype=self.args.torch_dtype)

        if('mistral' in self.args.model_name_or_path.lower()):
            self.gtemp=0.7
            self.top_p=1
        elif('llama' in self.args.model_name_or_path.lower()):
            self.gtemp=0.6
            self.top_p=0.9
        else:
            assert False, "Model not accepted"

        self.embedding_attr = 'model'
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f"Created {self.args.model_name_or_path}: {self.model.dtype} dtype")


        self.ltokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path, padding_side='left')
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path, padding_side='right')

        # import ipdb; ipdb.set_trace()
        if not(self.tokenizer.pad_token) and self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.ltokenizer.pad_token = self.ltokenizer.eos_token

        print('Set pad token to eos token: ' + self.tokenizer.pad_token)
        self.model.eval()



class GenEOL(torch.nn.Module):
    def __init__(self, accelerator, args) -> None:
        super().__init__()
        self.accelerator = accelerator
        self.llm = LLM(accelerator, args)
        self.args=args
        self.pooling_method = args.pooling_method
        self.encode_call=0



    def encode_queries(self, queries: Union[List[str], str], **kwargs) -> np.ndarray:
        """Used for encoding the queries of retrieval or reranking tasks"""
        return self.encode(queries, **kwargs)

    def encode_corpus(self, corpus: Union[List[str], str, List[Dict[str, str]]], **kwargs) -> np.ndarray:
        """Used for encoding the corpus of retrieval tasks"""
        if isinstance(corpus, dict):
            corpus = [corpus]
        if isinstance(corpus, list) and isinstance(corpus[0], dict):
            corpus = [
                doc["title"] + " " + doc["text"] if "title" in doc 
                else doc["text"] for doc in corpus
            ]
        return self.encode(corpus, **kwargs)



    @torch.no_grad()
    def encode(
        self,
        sentences: Union[List[str], str],
        convert_to_tensor: bool = False,
        args=None,
        **kwargs,
    ) -> np.ndarray:
        self.encode_call+=1
        
        input_was_string = False
        if isinstance(sentences, str):
            sentences = [sentences]
            input_was_string = True

        print(args.task, flush=True)
        # print(convert_to_tensor, "Convert to tensor", flush=True)
        
        # return np.zeros((len(sentences), 50))

        self.accelerator.wait_for_everyone()    
        start=time.time()
        
        with self.accelerator.split_between_processes(sentences) as sentences_rank_unchopped:
            np.random.seed(args.seed)
            self.llm.switchon_gen_model()
            all_embeddings = []
            
            save_path = os.path.join(args.output_folder, "..",'transformations')
            os.makedirs(save_path, exist_ok=True)
            print(save_path, flush=True)

            all_new_sentences_batch = []
            #! important to reduce size for all methods equally.
            # sentences_rank = self.llm.tokenizer.batch_decode(self.llm.tokenizer(sentences_rank_unchopped, max_length=args.max_length, truncation=True, add_special_tokens=False).input_ids)
            sentences_rank = sentences_rank_unchopped

            #! >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> PART 1
            if(not os.path.exists(f"{save_path}/{args.task}_sentences{self.encode_call}_{self.accelerator.process_index}.json")):
                print("process id", self.accelerator.process_index, flush=True)

                
                for start_index in tqdm(range(0, len(sentences_rank), args.batch_size), desc="Batches", disable=len(sentences_rank)<50):
                    
                    instructions_inputs_batch = []
                    sentences_batch = sentences_rank[start_index:start_index + args.batch_size]
                    if(args.method=='s5'):
                        for s in sentences_batch:
                            instructions_inputs_batch.extend([get_pos_prompt(1, s), get_pos_prompt(2, s), get_pos_prompt(3, s), get_pos_prompt(4, s)])
                        total_num_gens=args.num_gens*4

                        #! call LLM here
                        outputs = self.llm.generate(instructions_inputs_batch)

                        new_sentences_batch = []
                        for idx in range(len(sentences_batch)):
                            new_sentences_batch.append(sentences_batch[idx])
                            new_sentences_batch.extend(outputs[total_num_gens*idx:total_num_gens*(idx+1)])

                    elif(args.method=='d5'):
                        for s in sentences_batch:
                            instructions_inputs_batch.extend([get_diverse_prompt_fs(s)])

                        #! call LLM here
                        outputs = self.llm.generate(instructions_inputs_batch)
                        # outputs = [opt.strip().split("\n")[-1] for opt in outputs]

                        # new_sentences_batch = []
                        # for idx in range(len(sentences_batch)):
                        #     new_sentences_batch.append(sentences_batch[idx])
                        #     new_sentences_batch.extend(outputs[total_num_gens*idx:total_num_gens*(idx+1)])
                        total_num_gens=args.num_gens*10
                        new_sentences_batch = []
                        for idx in range(len(sentences_batch)):
                            new_sentences_batch.append(sentences_batch[idx])

                            gens = outputs[idx].split("\n")
                            # print(gens)
                            gens = [re.sub(r'^\d+\.\s*', '', gen).strip() for gen in gens if len(gen)>5]

                            # print("should not happen often", f"\n\n\n{outputs[idx]}\n\n\n", flush=True)
                            
                            if len(gens)<10:
                                gens += [new_sentences_batch[-1]]*(10-len(gens))
                                print("<<< lower length", flush=True)
                            elif len(gens)>10:
                                
                                gens = gens[:10]
                                print(">>> higher length", flush=True)
                            else:
                                pass
                            new_sentences_batch.extend(gens)
                    elif(args.method=='d52' or args.method=='c5' or args.method=='ch5'):
                        
                        outputs_extra = []
                        for sind, s in enumerate(sentences_batch):
                            instructions_inputs_batch.extend([get_task_specific_gen_prompt(s, args.task)])


                        #! call LLM here
                        outputs = self.llm.generate(instructions_inputs_batch)

                        # outputs = [opt.strip().split("\n")[-1] for opt in outputs]

                        # new_sentences_batch = []
                        # for idx in range(len(sentences_batch)):
                        #     new_sentences_batch.append(sentences_batch[idx])
                        #     new_sentences_batch.extend(outputs[total_num_gens*idx:total_num_gens*(idx+1)])
                        total_num_gens=args.num_gens*10
                        new_sentences_batch = []
                        for idx in range(len(sentences_batch)):
                            new_sentences_batch.append(sentences_batch[idx])
                            #! we use a hard coded value 10 here
                            if(len(sentences_batch[idx]) <= 1):
                                new_sentences_batch.extend([sentences_batch[idx]]*total_num_gens)
                                continue

                            try:
                                # import ipdb; ipdb.set_trace()   
                                gens = json_repair.loads(outputs[idx])
                                if(type(gens)==dict):
                                    gens = gens["generations"]
                                    gens = [gen.strip() for gen in gens if len(gen.strip())]
                                elif(type(gens)==list):
                                    gens = gens
                                else:
                                    print("text formattable but not dict, list", f"\n\n\n{gens}\n\n\n", flush=True)
                                
                            except:
                                gens = [sentences_batch[idx]]*total_num_gens
                                print("should not happen often", f"\n\n\n{outputs[idx]}\n\n\n", flush=True)
                            
                            if len(gens)<10:
                                gens += [new_sentences_batch[-1]]*(10-len(gens))
                                print("<<< lower length", flush=True)
                            elif len(gens)>10:
                                
                                gens = gens[:10]
                                print(">>> higher length", flush=True)
                            else:
                                pass
                            new_sentences_batch.extend(gens)
                    elif(args.method=='r5'):
                        for s in sentences_batch:
                            instructions_inputs_batch.extend([get_task_specific_gen_prompt(s, args.task)])

                        total_num_gens=1
                        
                        #! call LLM here
                        outputs = self.llm.generate(instructions_inputs_batch)
                        new_sentences_batch = []
                        for idx in range(len(sentences_batch)):
                            new_sentences_batch.append(sentences_batch[idx])
                            new_sentences_batch.extend(outputs[total_num_gens*idx:total_num_gens*(idx+1)])
                    elif(args.method=='b5'):
                        new_sentences_batch=sentences_batch
                        total_num_gens = 0
                    else:
                        assert False, "pick between s5 and b5"
                    all_new_sentences_batch.extend(new_sentences_batch)      
      

                with open(f"{save_path}/{args.task}_sentences{self.encode_call}_{self.accelerator.process_index}.json", "w") as f:
                    json.dump(all_new_sentences_batch, f)
            else:
                print("Loading first level generations", flush=True)
                with open(f"{save_path}/{args.task}_sentences{self.encode_call}_{self.accelerator.process_index}.json") as f:
                    all_new_sentences_batch = json.load(f)

                
                if(args.method=='s5'):
                    total_num_gens = 4*args.num_gens
                elif(args.method=='d5' or args.method=='d52'):
                    total_num_gens = 10*args.num_gens
                elif(args.method=='b5'):
                    total_num_gens = 0
                    # assert False, "b5 not compatibale with compositional"
                else:
                    assert False, "Not accepted method"




            #! >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> PART 2
            if(self.args.compositional and (not os.path.exists(f"{save_path}/{args.task}_compositional_sentences{self.encode_call}_{self.accelerator.process_index}.json"))):
                
                # print(len(all_new_sentences_batch), type(all_new_sentences_batch), all_new_sentences_batch[-1:])
                compinst_all_new_sentences = []
                orig_sentences = []
                for idx, s in enumerate(all_new_sentences_batch):
                    if(idx%(total_num_gens+1)==0):
                        orig_sentences.append(s)
                    else:
                        compinst_all_new_sentences.append(get_sum_prompt_fs(s))
            
                print(compinst_all_new_sentences)
                self.temp_num_gens = self.args.num_gens
                self.args.num_gens=1
                # for start_index in tqdm(range(0, len(comp_all_new_sentences), new_batch_size), desc="Batches", disable=len(comp_all_new_sentences_batch)<50):
                all_new_sentences_batch = []
                for start_index in tqdm(range(0, len(orig_sentences), args.batch_size), desc="Batches", disable=len(orig_sentences)<50):
                    sentences_batch = orig_sentences[start_index:start_index+args.batch_size]
                    compinst_sentences_batch = compinst_all_new_sentences[total_num_gens*start_index:total_num_gens*(start_index+args.batch_size)]                
                    
                    outputs = self.llm.generate(compinst_sentences_batch)
                    new_sentences_batch = []
                    for idx in range(len(sentences_batch)):
                        new_sentences_batch.append(sentences_batch[idx])
                        new_sentences_batch.extend(outputs[total_num_gens*idx:total_num_gens*(idx+1)])
                    all_new_sentences_batch.extend(new_sentences_batch)

                with open(f"{save_path}/{args.task}_compositional_sentences{self.encode_call}_{self.accelerator.process_index}.json", "w") as f:
                    json.dump(all_new_sentences_batch, f)

                self.args.num_gens=self.temp_num_gens
            elif(self.args.compositional):
                print("Loading compositional generations", flush=True)
                with open(f"{save_path}/{args.task}_compositional_sentences{self.encode_call}_{self.accelerator.process_index}.json") as f:
                    all_new_sentences_batch = json.load(f)
            else:
                pass
            

            print("Task specific Prompts" if args.tsep else "KWEOL Prompts", flush=True)   
            


            #! >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> PART 3
            if(not args.gen_only):
                self.llm.switchon_emb_model()          
                # all_new_sentences_batch = [f'<s>The essence of a sentence is often captured by its main subjects and actions, while descriptive terms provide additional but less central details. With this in mind , this sentence : "{s}" means in one word:"' for s in all_new_sentences_batch]

                all_new_sentences_batch = [get_task_specific_emb_prompt(s, args.task if args.tsep else None) for s in all_new_sentences_batch]
                print(all_new_sentences_batch[:5], len(all_new_sentences_batch), flush=True)


                
                # all_new_sentences_batch = [f'<s>This sentence : "{s}" means in one word:"' for s in all_new_sentences_batch]
                new_batch_size = (total_num_gens+1)*args.batch_size
                for start_index in tqdm(range(0, len(all_new_sentences_batch), new_batch_size), desc="Batches"):
                    new_sentences_batch = all_new_sentences_batch[start_index:start_index + new_batch_size]

                    # new_sentences_batch = [f'<s>This sentence : "{s}" means in one word:"' for s in new_sentences_batch]
                    #! embed here
                    last_hidden_state, inputs = self.llm.embed(new_sentences_batch)
                    
                    # if self.projection:
                    #     last_hidden_state = self.projection(last_hidden_state)

                    if ("mean" in args.pooling_method):
                        # Remove instruction tokens from the embeddings by masking them
                        for ii_idx, instruction_input in enumerate(new_sentences_batch):
                            insind = instruction_input.find('means in one word:"')+len('means in one word')
                            instruction_tokens = self.llm.tokenizer(instruction_input[:insind], add_special_tokens=False)["input_ids"]
                            inputs['attention_mask'][ii_idx, :len(instruction_tokens)] = 0

                    
                    # import ipdb; ipdb.set_trace()
                    embeddings = self.pooling(last_hidden_state, inputs['attention_mask'], recast=False).to('cpu')
                    if(torch.isnan(embeddings).any()):
                        import ipdb; ipdb.set_trace()

                    del last_hidden_state
                    del inputs

                    # Normalize can change the dtype (https://discuss.pytorch.org/t/tensor-in-float16-is-transformed-into-float32-after-torch-norm/110891)
                    if self.args.normalized: 
                        in_dtype = embeddings.dtype
                        embeddings = torch.nn.functional.normalize(embeddings, dim=-1).to(in_dtype)
                    embeddings = cast(torch.Tensor, embeddings)

                    try:
                        embeddings = torch.reshape(embeddings, (len(new_sentences_batch)//(total_num_gens+1),-1,embeddings.shape[-1]))
                        # assert embeddings.shape[-2]==5
                        embeddings = torch.mean(embeddings, -2)
                    except:
                        print("error params", len(new_sentences_batch), (total_num_gens+1), len(all_new_sentences_batch), new_batch_size, flush=True)

                    
                    all_embeddings.append(embeddings)

                all_embeddings = torch.cat(all_embeddings, dim=0)
            else:
                all_embeddings = torch.ones((len(sentences_rank_unchopped), 2))

        all_embeddings = [all_embeddings]
        all_embeddings=torch.cat(gather_object(all_embeddings), dim=0)

        all_embeddings = all_embeddings if convert_to_tensor else all_embeddings.cpu().to(torch.float32).numpy()


        return all_embeddings



    def pooling(
        self, hidden_state: torch.Tensor, attention_mask: torch.Tensor = None, recast: bool = False
    ) -> torch.Tensor:
        """
        Args:
            hidden_state: [b, n, d]
            attention_mask: [b, n]
        """
        # In case the model is distributed across multiple devices; hidden_state may end up on diff device
        hidden_state = hidden_state.to(attention_mask.device)
        if self.pooling_method == 'cls':
            embedding = hidden_state[:, 0]
        elif self.pooling_method == 'lasttoken':
            b, n, d = hidden_state.size()
            # Get the last `1` in the attention mask of each item
            # Often it is just `gather_indices = torch.argmin(attention_mask, 1, keepdim=False) - 1`
            # except when 1) There's all 1's 2) There's 0's before the 1's
            reversed_mask = torch.flip(attention_mask, dims=(1,))
            argmax_reverse = torch.argmax(reversed_mask, dim=1, keepdim=False)
            gather_indices = attention_mask.size(1) - argmax_reverse - 1
            # If there are empty sequences, where the index would become -1 it will crash so set them to 0
            gather_indices = torch.clamp(gather_indices, min=0)
            # Turn indices from shape [b] -> [b, 1, d]
            gather_indices = gather_indices.unsqueeze(-1).repeat(1, d)
            gather_indices = gather_indices.unsqueeze(1)
            assert gather_indices.shape == (b, 1, d)
            # Gather along the seq len: [b, n, d] -> [b, d]
            # Actually no need for the attention mask as we gather the last token where attn_mask=1 but
            # as some indices (which shouldn't be attended to) may be 0 due to clamp, use mask to ignore them again
            input_mask_expanded = attention_mask.unsqueeze(-1).expand((b, n, d)).float()
            embedding = torch.gather(hidden_state * input_mask_expanded, 1, gather_indices).squeeze(dim=1)
        elif self.pooling_method in ['mean', 'weightedmean']:
            if self.pooling_method == 'weightedmean':
                attention_mask *= attention_mask.cumsum(dim=1) # [0,1,1,1,0,0] -> [0,1,2,3,0,0]
            s = torch.sum(hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            embedding = s / d
        else: raise NotImplementedError(f"Unknown pooling method: {self.pooling_method}")
        # Recasting performs slightly worse but saves 50% space
        if recast: return embedding.to(hidden_state.dtype)
        return embedding

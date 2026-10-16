from datasets import DatasetDict, load_dataset
from typing import Dict, List
import re
import copy
import random

from data.base_builder import BaseBuilder
from data.triviaqa.env import TriviaQAEnv


TRIVIAQA_SYSTEM_PROMPT = """Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. \
"""

class TriviaQABuilder(BaseBuilder):  # Env

    def get_env_cls(self):
        return TriviaQAEnv

    def _build_sft_datasets(self) -> DatasetDict:
        
        # build train/valid dataset from agentbank
        train_ds = load_dataset("Solaris99/AgentBank", "triviaqa")["train"]
        
        valid_ratio = self.config.get("valid_ratio")
        all_size = len(train_ds)
        valid_size = int(all_size * valid_ratio)
        split = train_ds.train_test_split(test_size=valid_size, shuffle=True)
        raw_train_dataset, raw_valid_dataset = split["train"], split["test"]

        # build test dataset from triviaqa
        ds = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia.nocontext")   
        raw_test_dataset = ds["validation"]  

        # preprocess
        num_workers = 32
        train_dataset = raw_train_dataset.map(self._sft_preprocess, num_proc=num_workers).select_columns(self._sft_keep_keys())
        valid_dataset = raw_valid_dataset.map(self._sft_preprocess, num_proc=num_workers).select_columns(self._sft_keep_keys())
        test_dataset = raw_test_dataset.map(self._rl_preprocess, num_proc=num_workers).select_columns(self._rl_keep_keys())
        
        dataset_dict = DatasetDict()
        dataset_dict["train"] = train_dataset
        dataset_dict["valid"] = valid_dataset
        dataset_dict["test"] = test_dataset

        return dataset_dict

    def _build_rl_datasets(self) -> DatasetDict:
        
        ds = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia.nocontext")   
        # Wikipedia has no labeled test split. Hold out official validation for
        # evaluation and split official train by question to prevent overlap.
        raw_test_dataset = ds["validation"]
        heldout = {row["question"].strip().casefold() for row in raw_test_dataset}
        grouped = {}
        for index, row in enumerate(ds["train"]):
            question = row["question"].strip().casefold()
            if question not in heldout:
                grouped.setdefault(question, index)
        questions = sorted(grouped)
        random.Random(self.config.get("seed", 42)).shuffle(questions)
        valid_size = max(1, int(len(questions) * self.config.get("valid_ratio", 0.1)))
        valid_questions, train_questions = questions[:valid_size], questions[valid_size:]
        train_size = self.config.get("train_size", 0)
        if train_size:
            train_questions = train_questions[:train_size]
        raw_train_dataset = ds["train"].select([grouped[q] for q in train_questions])
        raw_valid_dataset = ds["train"].select([grouped[q] for q in valid_questions])
       
        num_workers = 32
        train_dataset = raw_train_dataset.map(self._rl_preprocess, num_proc=num_workers).select_columns(self._rl_keep_keys())
        valid_dataset = raw_valid_dataset.map(self._rl_preprocess, num_proc=num_workers).select_columns(self._rl_keep_keys())
        test_dataset = raw_test_dataset.map(self._rl_preprocess, num_proc=num_workers).select_columns(self._rl_keep_keys())

        dataset_dict = DatasetDict()
        dataset_dict["train"] = train_dataset
        dataset_dict["valid"] = valid_dataset
        dataset_dict["test"] = test_dataset

        return dataset_dict

    @classmethod
    def _sft_preprocess(cls, example: Dict):

        def _add_user_special_tokens(content: str) -> str:
            observation_match = re.search(r'Observation: (.*)', content)

            if observation_match:
                observation_content = f"<observation> {observation_match.group(1).strip()} </observation>"
            else:
                observation_content = content
            
            return observation_content

        def _add_assistant_special_tokens(content: str) -> str:
            thought_match = re.search(r'Thought: (.*?)(?=\nAction:|\nFinal Answer:|$)', content, re.DOTALL)
            action_match = re.search(r'Action: search\[(.*?)\]', content)
            answer_match = re.search(r'Final Answer: (.*)', content)

            parts = []
            
            if thought_match:
                thought_content = thought_match.group(1).strip()
                parts.append(f"<think> {thought_content} </think>")
            
            if action_match:
                action_content = action_match.group(1).strip()
                parts.append(f"<search> {action_content} </search>")

            if answer_match:
                answer_content = answer_match.group(1).strip()
                parts.append(f"<answer> {answer_content} </answer>")
            
            aggregated_content = "\n".join(parts)
            return aggregated_content

        messages = []
        system_prompt = {"role": "system", "content": TRIVIAQA_SYSTEM_PROMPT.strip()}
        messages.append(system_prompt)

        for sample in example["conversations"]:
            message = {}
            # role
            if sample["from"] == "human":
                message["role"] = "user"
                message["content"] = _add_user_special_tokens(sample["value"])
            elif sample["from"] == "gpt":
                message["role"] = "assistant"
                message["content"] = _add_assistant_special_tokens(sample["value"])
            else:
                raise ValueError("Unsupported Role type.")
            
            messages.append(message)
        
        return {
            "messages": messages
        }

    @classmethod
    def _sft_keep_keys(cls) -> List[str]:
        return ["messages"]

    @classmethod
    def _rl_preprocess(cls, example: Dict) -> Dict:
        output = copy.deepcopy(example)
        output["answer"] = output["answer"]["normalized_aliases"]
        output["prompt"] = output["question"]
        return output

    @classmethod
    def _rl_keep_keys(cls) -> List[str]:
        return ["prompt", "answer"]

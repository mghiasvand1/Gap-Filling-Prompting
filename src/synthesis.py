from openai import OpenAI
from tqdm import tqdm
import pandas as pd
import json
import ast

api_key = "your_api_key"
client = OpenAI(api_key=api_key)


def construct_prompt(question, solution):
    prompt = """You will receive a grade-school math question along with its solution. Your task is to act as an experienced teacher by providing essential, complete, yet concise hints that guide the student through the key steps of the solution. Avoid including hints that are easily understood or simply restate the purpose already mentioned in the question, even in a different form, as this would be redundant. Keep the number of hints to a minimum. Then, to create the modified version, integrate the summarized version of each hint into the most relevant sentence of the question, without creating a separate sentence for it and avoiding unifying or grouping all the hints together in one place. Next, act as a professional writer and rephrase the modified version into a continuous, well-structured, complete, yet concise instruction that includes all the initial numerical assumptions of the question, without introducing any additional numbers or directly mentioning any part of the answer. The hints must be seamlessly integrated into the sentences and must not stand out as separate guides. Do not include any computational operations or numerical calculations in either the hints or the instruction. Finally, provide Python code that accurately implements your instruction, ensuring the final answer is stored in the 'result' variable without including any comments in your code. Return your hints, instruction, and code explicitly in the following JSON format: {"hints": [], "instruction": "", "code": ""}.\n\n""" + f"""[Question]\n{question}\n\n[Solution]\n{solution}"""
    return prompt


def get_model_response(prompt):
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0,
        max_tokens=1024,
        response_format={"type": "json_object"}
    )
    response = completion.choices[0].message.content.strip()
    return response


with open('/content/drive/MyDrive/data/GSM8K/train.json') as f:
    data = json.load(f)
df = pd.DataFrame(data)[["question", "answer", "num_answer"]]
df = df.rename(columns={"question": "input"})

df['target'] = ""
_inputs, instructions, _code, num_answers = [], [], [], []

for index, row in tqdm(df.iterrows(), total=len(df)):
    _input = row['input']
    solution = row['answer']
    prompt = construct_prompt(_input, solution)
    try:
        response = ast.literal_eval(get_model_response(prompt))
        hints = response["hints"]
        _inputs.append(f"{_input} ## "+" & ".join(hints))
        hints = " & ".join(hints)
        instructions.append(response["instruction"])
        _code.append(response["code"])
        num_answers.append(row['num_answer'])
        df.at[index, 'target'] = hints
    except:
        df = df.drop(index)

hints_df = df

df = pd.DataFrame({
    'input': _inputs,
    'target': instructions,
    'code': _code,
    'num_answer': num_answers
})
instruction_df = df

code_df = df

rows_to_keep = []
for index, row in code_df.iterrows():
    local_vars = {}
    try:
        exec(row['code'], {}, local_vars)
        if 'result' in local_vars:
            result = local_vars['result']
            if float(result) == float(row['num_answer']):
                rows_to_keep.append(index)
    except (Exception, SyntaxError, NameError) as e:
        pass

hints_df = hints_df.drop(columns=['answer', 'num_answer'])
hints_df = hints_df.loc[rows_to_keep].reset_index(drop=True)
hints_df.to_csv("/content/drive/MyDrive/data/GSM8K/hints.csv", index=False)

instruction_df = instruction_df.drop(columns=['code', 'num_answer'])
instruction_df = instruction_df.loc[rows_to_keep].reset_index(drop=True)
instruction_df.to_csv('/content/drive/MyDrive/data/GSM8K/instruction.csv', index=False)

code_df = code_df.drop(columns=['target', 'num_answer'])
code_df = code_df.loc[rows_to_keep].reset_index(drop=True)
code_df = code_df.rename(columns={"code": "target"})
code_df.to_csv("/content/drive/MyDrive/data/GSM8K/code.csv", index=False)

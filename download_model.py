from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# # 指定模型名称
# model_name = "microsoft/phi-1"

# # 下载并加载模型和分词器
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# 模型现在可以使用

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "microsoft/phi-1"
save_directory = "home/u2024140809/share/models--microsoft--phi-1"  # 替换为你想要保存模型的路径
config_model="../share/models--microsoft--phi-1/models--microsoft--phi-1/snapshots/b9ac0e6d78d43970ecf88e9e0154b3a7da20ed89"

config = AutoConfig.from_pretrained(config_model)
print(config)
# tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1", cache_dir=save_directory)
# model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1", cache_dir=save_directory)

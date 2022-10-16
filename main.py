# This flag is the difference between SQUAD v1 or 2 (if you're using another dataset, it indicates if impossible
# answers are allowed or not).
squad_v2 = False
model_checkpoint = "distilbert-base-uncased"
batch_size = 16

from datasets import load_dataset, load_metric
from icecream import ic
from datasets import ClassLabel, Sequence
import random
import pandas as pd
from IPython.display import display, HTML
from transformers import AutoTokenizer
import transformers


datasets = load_dataset("squad_v2" if squad_v2 else "squad")
ic(datasets)
ic(datasets["train"][0])


def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(
        dataset
    ), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset) - 1)
        while pick in picks:
            pick = random.randint(0, len(dataset) - 1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
        elif isinstance(typ, Sequence) and isinstance(typ.feature, ClassLabel):
            df[column] = df[column].transform(
                lambda x: [typ.feature.names[i] for i in x]
            )
    ic(df)


show_random_elements(datasets["train"])


# get a tokenizer that corresponds to the model architecture we want to use,
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

ic(tokenizer("What is your name?", "My name is Sylvain."))
# Now one specific thing for the preprocessing in question answering is how to deal with very long documents.
# We usually truncate them in other tasks, when they are longer than the model maximum sentence length,
# but here, removing part of the the context might result in losing the answer we are looking for.
# To deal with this, we will allow one (long) example in our dataset to give several input features,
# each of length shorter than the maximum length of the model (or the one we set as a hyper-parameter).
# Also, just in case the answer lies at the point we split a long context, we allow some overlap between
# the features we generate controlled by the hyper-parameter doc_stride:

max_length = 384  # The maximum length of a feature (question and context)
doc_stride = 128  # The allowed overlap between two part of the context when splitting is performed.

for i, example in enumerate(datasets["train"]):
    if len(tokenizer(example["question"], example["context"])["input_ids"]) > 384:
        break
    example = datasets["train"][i]

# Without any truncation, we get the following length for the input IDs:
ic(len(tokenizer(example["question"], example["context"])["input_ids"]))


# with truncation, we will lose information (and possibly the answer to our question):

ic(len(
    tokenizer(
        example["question"],
        example["context"],
        max_length=max_length,
        truncation="only_second",
    )["input_ids"]
))
# Note that we never want to truncate the question, only the context,
# and so we use the only_second truncation method. Our tokenizer can
# automatically return a list of features capped by a certain maximum length,
# with the overlap we talked about above, we just have to tell it to do
# so with return_overflowing_tokens=True and by passing the stride:


tokenized_example = tokenizer(
    example["question"],
    example["context"],
    max_length=max_length,
    truncation="only_second",
    return_overflowing_tokens=True,
    stride=doc_stride,
)
ic(tokenized_example)

# Now we don't have one list of input_ids, but several:
ic([len(x) for x in tokenized_example["input_ids"]])

for x in tokenized_example["input_ids"][:2]:
    print(tokenizer.decode(x))


# It's going to take some work to properly label the answers here:
# we need to find in which of those features the answer actually is,
# and where exactly in that feature. The models we will use require
# the start and end positions of these answers in the tokens,
# so we will also need to map parts of the original context to
# some tokens. Thankfully, the tokenizer we're using can help us
# with that by returning an offset_mapping:


tokenized_example = tokenizer(
    example["question"],
    example["context"],
    max_length=max_length,
    truncation="only_second",
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
    stride=doc_stride,
)
print(tokenized_example["offset_mapping"][0][:100])
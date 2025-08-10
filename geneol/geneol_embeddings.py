from geneol import GenEOL
from accelerate import Accelerator, InitProcessGroupKwargs
from argparse import Namespace


def get_geneol_embeddings(sentences):
    from accelerate import Accelerator, InitProcessGroupKwargs
    from argparse import Namespace
    from geneol import GenEOL

    args = Namespace(
        model_name_or_path="mistralai/Mistral-7B-v0.1",
        gen_model_name_or_path="mistralai/Mistral-7B-Instruct-v0.1",
        method="s5",
        batch_size=16,
        num_gens=2,
        max_length=128,
        pooling_method="mean",
        suffix="custom",
        output_folder="./resultsTF/custom",
        compositional=True,
        task="custom",
        seed=42,
        normalized=True,
        penultimate_layer=-1,
        tsep=False,
        gen_only=False,
        torch_dtype="bfloat16", 
    )
    init_proc = InitProcessGroupKwargs(timeout=1000)
    accelerator = Accelerator(kwargs_handlers=[init_proc])
    model = GenEOL(accelerator=accelerator, args=args)
    embeddings = model.encode(sentences, args=args)
    return embeddings

# sentences = ["This is the embarrassing moment a Buckingham Palace guard slipped and fell on a manhole cover in front of hundreds of shocked tourists as he took up position in his sentry box. The Queen's Guard was left red-faced after the incident."]

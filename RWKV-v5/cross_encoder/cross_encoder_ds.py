# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
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

# Lint as: python3
"""mMARCO dataset."""

import datasets
import os

_CITATION = """
@misc{bonifacio2021mmarco,
      title={mMARCO: A Multilingual Version of the MS MARCO Passage Ranking Dataset},
      author={Luiz Henrique Bonifacio and Israel Campiotti and Vitor Jeronymo and Hugo Queiroz Abonizio and Roberto Lotufo and Rodrigo Nogueira},
      year={2021},
      eprint={2108.13897},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

_URL = "https://github.com/unicamp-dl/mMARCO"

_DESCRIPTION = """
mMARCO translated datasets
"""


def generate_examples_triples(filepath, collection_path, queries_path):
    collection = {}
    with open(collection_path, encoding="utf-8") as f:
        for line in f:
            doc_id, doc = line.rstrip().split("\t")
            collection[doc_id] = doc

    queries = {}
    with open(queries_path, encoding="utf-8") as f:
        for line in f:
            query_id, query = line.rstrip().split("\t")
            queries[query_id] = query

    with open(filepath, encoding="utf-8") as f:
        for (idx, line) in enumerate(f):
            query_id, pos_id, neg_id = line.rstrip().split("\t")
            features = {
                "query": queries[query_id],
                "positive": collection[pos_id],
                "negative": collection[neg_id],
            }
            yield idx, features


def generate_examples_tuples(filepath):
    with open(filepath, encoding="utf-8") as f:
        for (idx, line) in enumerate(f):
            idx, text = line.rstrip().split("\t")
            features = {
                "id": idx,
                "text": text,
            }
            yield idx, features


def generate_examples_runs(filepath, collection_path, queries_path):
    collection = {}
    with open(collection_path, encoding="utf-8") as f:
        for line in f:
            doc_id, doc = line.rstrip().split("\t")
            collection[doc_id] = doc

    queries = {}
    with open(queries_path, encoding="utf-8") as f:
        for line in f:
            query_id, query = line.rstrip().split("\t")
            queries[query_id] = query

    qid_to_ranked_candidate_passages = {}
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            qid, pid, rank = line.rstrip().split("\t")
            if qid not in qid_to_ranked_candidate_passages:
                qid_to_ranked_candidate_passages[qid] = []
            qid_to_ranked_candidate_passages[qid].append(pid)

    for (idx, qid) in enumerate(qid_to_ranked_candidate_passages):
        features = {
            "id": qid,
            "query": queries[qid],
            "passages": [
                {
                    "id": pid,
                    "passage": collection[pid],
                }
                for pid in qid_to_ranked_candidate_passages[qid]
            ],
        }
        yield idx, features


_BASE_URLS = {
    "collections": "https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/collections/",
    "queries-train": "https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/queries/train/",
    "queries-dev": "https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/queries/dev/",
    "runs": "https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/runs/",
    "train": "https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/triples.train.ids.small.tsv",
}

LANGUAGES = [
    "arabic",
    "chinese",
    "dutch",
    "english",
    "french",
    "german",
    "hindi",
    "indonesian",
    "italian",
    "japanese",
    "portuguese",
    "russian",
    "spanish",
    "vietnamese",
]


class MMarco(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = (
        [
            datasets.BuilderConfig(
                name=language,
                description=f"{language.capitalize()} triples",
                version=datasets.Version("2.0.0"),
            )
            for language in LANGUAGES
        ]
        + [
            datasets.BuilderConfig(
                name=f"collection-{language}",
                description=f"{language.capitalize()} collection version v2",
                version=datasets.Version("2.0.0"),
            )
            for language in LANGUAGES
        ]
        + [
            datasets.BuilderConfig(
                name=f"queries-{language}",
                description=f"{language.capitalize()} queries version v2",
                version=datasets.Version("2.0.0"),
            )
            for language in LANGUAGES
        ]
        + [
            datasets.BuilderConfig(
                name=f"runs-{language}",
                description=f"{language.capitalize()} runs version v2",
                version=datasets.Version("2.0.0"),
            )
            for language in LANGUAGES
        ]
    )

    DEFAULT_CONFIG_NAME = "english"

    def _info(self):
        name = self.config.name
        if name.startswith("collection") or name.startswith("queries"):
            features = {
                "id": datasets.Value("int32"),
                "text": datasets.Value("string"),
            }
        elif name.startswith("runs"):
            features = {
                "id": datasets.Value("int32"),
                "query": datasets.Value("string"),
                "passages": datasets.Sequence(
                    {
                        "id": datasets.Value("int32"),
                        "passage": datasets.Value("string"),
                    }
                ),
            }
        else:
            features = {
                "query": datasets.Value("string"),
                "positive": datasets.Value("string"),
                "negative": datasets.Value("string"),
            }

        return datasets.DatasetInfo(
            description=f"{_DESCRIPTION}\n{self.config.description}",
            features=datasets.Features(features),
            supervised_keys=None,
            homepage=_URL,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        print("data_dir",self.config.data_dir)
        data_dir = self.config.data_dir
        if data_dir is None:
            if self.config.name.startswith("collection"):
                url = _BASE_URLS["collections"] + self.config.name[11:] + "_collection.tsv"
                dl_path = dl_manager.download_and_extract(url)
                return (datasets.SplitGenerator(name="collection", gen_kwargs={"filepath": dl_path}),)
            elif self.config.name.startswith("queries"):
                urls = {
                    "train": _BASE_URLS["queries-train"] + self.config.name[8:] + "_queries.train.tsv",
                    "validation": _BASE_URLS["queries-dev"] + self.config.name[8:] + "_queries.dev.tsv",
                    "dev": _BASE_URLS["queries-dev"] + self.config.name[8:] + "_queries.dev.small.tsv",
                }
                dl_path = dl_manager.download_and_extract(urls)
                return [
                    datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": dl_path["train"]}),
                    datasets.SplitGenerator(name="dev.full", gen_kwargs={"filepath": dl_path["validation"]}),
                    datasets.SplitGenerator(name="dev", gen_kwargs={"filepath": dl_path["dev"]}),
                ]
            elif self.config.name.startswith("runs"):
                urls = {
                    "collection": _BASE_URLS["collections"] + self.config.name[5:] + "_collection.tsv",
                    "queries": _BASE_URLS["queries-dev"] + self.config.name[5:] + "_queries.dev.tsv",
                    "run": _BASE_URLS["runs"] + "run.bm25_" + self.config.name[5:] + "-msmarco.txt",
                }

                dl_path = dl_manager.download_and_extract(urls)
                return (
                    datasets.SplitGenerator(
                        name="bm25",
                        gen_kwargs={
                            "filepath": dl_path["run"],
                            "args": {
                                "collection": dl_path["collection"],
                                "queries": dl_path["queries"],
                            },
                        },
                    ),
                )
            else:
                urls = {
                    "collection": _BASE_URLS["collections"] + self.config.name + "_collection.tsv",
                    "queries": _BASE_URLS["queries-train"] + self.config.name + "_queries.train.tsv",
                    "train": _BASE_URLS["train"],
                }
                dl_path = dl_manager.download_and_extract(urls)

                return [
                    datasets.SplitGenerator(
                        name=datasets.Split.TRAIN,
                        gen_kwargs={
                            "filepath": dl_path["train"],
                            "args": {
                                "collection": dl_path["collection"],
                                "queries": dl_path["queries"],
                            },
                        },
                    )
                ]
        else:
            # New code for reading files from a directory
            if self.config.name.startswith("collection"):
                filepath = os.path.join(data_dir, self.config.name[11:] + "_collection.tsv")
                return (datasets.SplitGenerator(name="collection", gen_kwargs={"filepath": filepath}),)
            elif self.config.name.startswith("queries"):
                filepaths = {
                    "train": os.path.join(data_dir, self.config.name[8:] + "_queries.train.tsv"),
                    "validation": os.path.join(data_dir, self.config.name[8:] + "_queries.dev.tsv"),
                    "dev": os.path.join(data_dir, self.config.name[8:] + "_queries.dev.small.tsv"),
                }
                return [
                    datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": filepaths["train"]}),
                    datasets.SplitGenerator(name="dev.full", gen_kwargs={"filepath": filepaths["validation"]}),
                    datasets.SplitGenerator(name="dev", gen_kwargs={"filepath": filepaths["dev"]}),
                ]
            elif self.config.name.startswith("runs"):
                filepaths = {
                    "collection": os.path.join(data_dir, self.config.name[5:] + "_collection.tsv"),
                    "queries": os.path.join(data_dir, self.config.name[5:] + "_queries.dev.tsv"),
                    "run": os.path.join(data_dir, "run.bm25_" + self.config.name[5:] + "-msmarco.txt"),
                }
                return (
                    datasets.SplitGenerator(
                        name="bm25",
                        gen_kwargs={
                            "filepath": filepaths["run"],
                            "args": {
                                "collection": filepaths["collection"],
                                "queries": filepaths["queries"],
                            },
                        },
                    ),
                )
            else:
                filepaths = {
                    "collection": os.path.join(data_dir, self.config.name + "_collection.tsv"),
                    "queries": os.path.join(data_dir, self.config.name + "_queries.train.tsv"),
                    "train": os.path.join(data_dir, "triples.train.ids.small.tsv"),
                }
                print("filepaths",filepaths)
                return [
                    datasets.SplitGenerator(
                        name=datasets.Split.TRAIN,
                        gen_kwargs={
                            "filepath": filepaths["train"],
                            "args": {
                                "collection": filepaths["collection"],
                                "queries": filepaths["queries"],
                            },
                        },
                    )
                ]

    def _generate_examples(self, filepath, args=None):
        """Yields examples."""

        if self.config.name.startswith("collection") or self.config.name.startswith("queries"):
            return generate_examples_tuples(filepath)
        if self.config.name.startswith("runs"):
            return generate_examples_runs(filepath, args["collection"], args["queries"])
        else:
            return generate_examples_triples(filepath, args["collection"], args["queries"])
        


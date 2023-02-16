# %%
import numpy as np
import rust_circuit as rc

from rust_circuit.causal_scrubbing.experiment import (
    Dataset,
    Experiment,
)
from rust_circuit.causal_scrubbing.hypothesis import CondSampler, Correspondence, InterpNode
import torch
import attrs
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
from rust_circuit.causal_scrubbing.hypothesis import (
    CondSampler,
    Correspondence,
    InterpNode,
    chain_excluding,
    corr_root_matcher,
)

import rust_circuit as rc

# initialisation of constant values for placeholders


# for IOI sequences
IO_TOK = -1
S_TOK = -2

# for ABC sequences
N1_TOK = -3
N2_TOK = -4

EMPTY_TOK = -6

EMPTY_LABEL = -42


# %% [markdown]
# In the previous example of causal scrubbing, we used a `CondSampler` as a filter.
# Given a reference dataset `ref` and a random dataset `ds`, we pick samples from `ds` that
# matches the samples from `ref`.

# However, sometimes the criteria implemented by your `CondSampler` is so restrictive that for a given x in `ref`,
# is only a few samples in `ds` that matches x. Seomtimes there is only one: this is x itself!

# %% [markdown]

# TODO add an example with a warning because of not enough samples

# %% [markdown]

# For instance if we're trying to find sequences that matche the condition "same indirect object"
# for x="Alice and Bob went to the store, Alice gave a drink to" in `ref`. If the number of possible names is of the same order
# than the size of our dataset, it is possible x is the only sequence that has "Bob" as indirect object. This is
# unfortnate, as we're not scrubbing any information here.
# However, there is a much more efficient way to deal with this: instead of filtering `ds` to find sentences
# with Bob as indirect object, we can easily _generate_on the fly such sentences.

# To deal with this, instead of filtering `ds` according to a condition, we will generate a dataset `ds` that matches
# this condition.


# %% [markdown]
#
# To do this, we'll define a custom Dataset object that will be created by a IOIDataSource object.
# * IOIDataSource is in charge of generating the data according ot a set of constraints.
# * The IOICSDataset object is stroing the data generated by the datasource. It keeps in memory
#  the datasource it has been generated from. It is a subclass of the causal scrubbing Dataset object
# and as such is a frozen class.

# We'll first have a look at the IOIDataSource object, then at the IOIDataSource object.

# %% [markdown]

# The dataset generator class. It's storing information about the specificities
# of the dataset we want to generate (e.g. the templates, the set of possible names, etc.) and
# create as many sample as needed (in the `gen` method).

# IOIDataSource only deals with tokenized data, the template it receive in __init__ are already tokenized
# and the names are tokenized as well.

# It can also be called with additionnal constraints, e.g. on the order of names, on the value of names
# at particular position etc. by using `gen_with_contraints`

# TBD: I'm not concerned about spelling errors in the text as I can fix those easily, but can you refactor gen_with_contraints -> gen_with_constraints in the code?

# %% [markdown]

# Useful definitions:
# * A example of **template** is "A and B went to the store, B gave a drink to" (A and B are palceholders for names).
# It contains fixed values for the place and the object given
#     * For ABC sequences, the placeholders are the variables `N1_TOK`, `N2_TOK` and `S_TOK` (in this order)
#     * For IOI sequences, the placeholders are the variables `IO_TOK` and `S_TOK` (the order varies with the template)
# * There are two **template families**: "IOI" and "ABC"
# * An example of **position** is "ABB", the order of the names in the template
# * `names_in_order` store the tokenized names in the order they appear in the sequences.
# * `io_tokens` store the tokenized IO for each sentence. For ABC, this is chosen the _first_ indirect object.
# * `s_tokens` store the subject token (not ambiguity between ABC and IOI here)
# * `io2_tokens` stores the tokenized second IO for ABC sequences only. For IOI it's `EMPTY_TOK`.
# %% [markdown]

# `gen_with_contraints` takes in argument a list of templates (one per future sequence) and three lists of names, `n1_names, n2_names, n3_names`.
# No more randomness appear here and the number of sample to generate is fixed This function is in charge of putting the right names at the
# correct place inside the templates.

# %% [markdown]

# Then we can define a list of methods that are the one used to generate dataset on the fly.
# e.g. `gen_for_first_name` generate a dataset while keeping the first name constant no matter its role, no matter
# the template family.
# `gen_for_io_and_position` generates a dataset with the same IO (or pair of IO for ABC), and keep the same position of the IO in the sentences.

# TBD: template is tensor here? add typing
def get_placeholder_order(template, return_idx=False):
    """Return the placeholder name value in the order they appear in the template"""
    if IO_TOK in template:
        sorted_idx = torch.cat([torch.where(template == IO_TOK)[0], torch.where(template == S_TOK)[0]]).sort().values
        if return_idx:
            return sorted_idx
        else:
            return tuple(template[sorted_idx].tolist())
    else:
        sorted_idx = torch.cat(
            [torch.where(template == N1_TOK)[0], torch.where(template == N2_TOK)[0], torch.where(template == S_TOK)[0]]
        )
        if return_idx:
            return sorted_idx
        else:
            return (N1_TOK, N2_TOK, S_TOK)


# TBD: what is template here? add typing
def get_template_family(template):
    return IO_TOK in template


# %% The definition of the `IOIDataSource`


class IOIDataSource:  # only tokenized information here
    # put tokenizer here
    def __init__(self, all_templates: torch.Tensor, all_names: torch.Tensor, tokenizer: GPT2TokenizerFast):
        self.tokenizer = tokenizer
        self.all_templates = all_templates  # we only deal with aligned sequences, we don't have any padding here
        # each prototype sentence gives three templates: ABC, ABB, BAB

        self.all_names = all_names  # 1D tensor containing all the tokenized names
        self.rng: np.random.Generator = np.random.default_rng()

    def gen_templates(self, num, templates_to_match=None, match_position=False, match_family=False):
        """ "If match_position is True, we will filter templates to make the name order match
        the orders in templates_to_match. If match_family is True, we will make the template famility
        match the templates_to_match"""
        if templates_to_match is None:
            templates = torch.tensor(self.rng.choice(self.all_templates, num, replace=True))
        else:
            if match_family:
                matching_fn = get_template_family
            elif match_position:
                matching_fn = get_placeholder_order
            else:
                raise ValueError("You need to specify a matching function")

            nb = 0
            templates = torch.zeros((num, len(self.all_templates[0])))
            while nb < num:
                template = torch.tensor(self.rng.choice(self.all_templates, 1, replace=True))[0]
                if matching_fn(template) == matching_fn(templates_to_match[nb]):
                    templates[nb] = template.clone()
                    nb += 1

        return templates

    def gen(self, num):
        """Generate the data to create a IOICSDataset without particular constraints"""
        templates = self.gen_templates(num)
        names = torch.tensor(self.rng.choice(self.all_names, (num, 3), replace=True))

        return self.gen_with_contraints(templates, names[:, 0], names[:, 1], names[:, 2], num)

    def gen_with_contraints(self, templates, n1_names, n2_names, n3_names, num, enforce_order=False):
        """
        Generate the data to create a IOICSDataset
        If enforce_order is True, then the names are interpreted as the order in the sequence
        Else the names are interpreted as their semantic role
        """
        if enforce_order:
            names_in_order = torch.cat(
                [torch.unsqueeze(n1_names, 1), torch.unsqueeze(n2_names, 1), torch.unsqueeze(n3_names, 1)], dim=1
            )
            io_tokens = torch.zeros((num)).long()
            s_tokens = torch.zeros((num)).long()
            io2_tokens = torch.zeros((num)).long()
        else:  # we interpret the names in arguments as their role in the sentence instead of their order
            io_tokens = n1_names.long()
            s_tokens = n2_names.long()
            io2_tokens = n3_names.long()
            names_in_order = torch.zeros((num, 3))

        tokens = templates.clone().long()
        labels = torch.zeros((num, 2))
        for i, template in enumerate(templates):
            if IO_TOK in template:  # IOI seq

                if enforce_order:  # we replace according to the position in the sentence
                    if get_placeholder_order(template) == (IO_TOK, S_TOK, S_TOK):
                        io_tokens[i] = n1_names[i]
                        s_tokens[i] = n2_names[i]

                    elif get_placeholder_order(template) == (S_TOK, IO_TOK, S_TOK):
                        s_tokens[i] = n1_names[i]
                        io_tokens[i] = n2_names[i]
                    else:
                        raise ValueError("Invalid template")

                io2_tokens[i] = EMPTY_TOK
                labels[i, 0] = io_tokens[i]
                labels[i, 1] = EMPTY_LABEL
                tokens[i, torch.where(template == IO_TOK)] = io_tokens[i]
                tokens[i][torch.where(template == S_TOK)] = s_tokens[i]
                names_in_order[i] = tokens[i][get_placeholder_order(template, return_idx=True).tolist()]

            else:  # ABC seq, same behavior no matter if we enforce order or not
                if enforce_order:
                    io_tokens[i] = n1_names[i]
                    io2_tokens[i] = n2_names[i]
                    s_tokens[i] = n3_names[i]

                tokens[i, torch.where(template == N1_TOK)] = io_tokens[i]
                tokens[i, torch.where(template == N2_TOK)] = io2_tokens[i]
                tokens[i, torch.where(template == S_TOK)] = s_tokens[i]
                labels[i, 0] = io_tokens[i]
                labels[i, 1] = io2_tokens[i]
                names_in_order[i] = torch.tensor([io_tokens[i], io2_tokens[i], s_tokens[i]])

        sequences_types = torch.tensor([1 if IO_TOK in template else 0 for template in templates])
        string_sentences = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
        arrs = (rc.Array(tokens.to(torch.int64), name="tokens"), rc.Array(labels.float(), name="labels"))
        return (
            arrs,
            tokens.to(torch.int64),
            labels,
            io_tokens,
            s_tokens,
            io2_tokens,
            templates,
            sequences_types,
            string_sentences,
            names_in_order,
            self,
            num,
        )

    def gen_for_first_name(self, names_in_order, num, ABC_IOI_swap=True):
        """keep the same first name no matter its role"""
        templates = self.gen_templates(num, match_family=ABC_IOI_swap)
        names = torch.tensor(self.rng.choice(self.all_names, (num, 3), replace=True))

        return self.gen_with_contraints(
            templates, names_in_order[:, 0], names[:, 1], names[:, 2], num, enforce_order=True
        )

    def gen_for_sec_name(self, names_in_order, num, ABC_IOI_swap=True):
        """keep the same first name no matter its role"""
        templates = self.gen_templates(num, match_family=ABC_IOI_swap)
        names = torch.tensor(self.rng.choice(self.all_names, (num, 3), replace=True))

        return self.gen_with_contraints(
            templates, names[:, 0], names_in_order[:, 1], names[:, 2], num, enforce_order=True
        )

    def gen_for_third_name(self, names_in_order, num, ABC_IOI_swap=True):
        """keep the same first name no matter its role"""
        templates = self.gen_templates(num, match_family=ABC_IOI_swap)
        names = torch.tensor(self.rng.choice(self.all_names, (num, 3), replace=True))

        return self.gen_with_contraints(
            templates,
            names[:, 0],
            names_in_order[:, 2],  # enforcing the third name is equivalent to enforcing the subject
            names[:, 2],
            num,
        )

    def gen_for_io_and_position(self, io_names, templates, num):
        """Generate a dataset with:
        * sequence with the same name order and the same IO token as io_names for sequences in IOI
        * random ABC sequences for sequences in ABC"""
        pos_matching_templates = self.gen_templates(num, templates_to_match=templates, match_position=True)
        names = torch.tensor(self.rng.choice(self.all_names, (num, 3), replace=True))
        sequences_types = torch.tensor([1 if IO_TOK in template else 0 for template in templates])
        io_for_ioi = io_names * sequences_types + names[:, 0] * (
            1 - sequences_types
        )  # we keep the io_name for IOI seq, random name for ABC
        return self.gen_with_contraints(pos_matching_templates, io_for_ioi, names[:, 1], names[:, 2], num)

    def gen_for_io(self, io_names, ordered_names, templates, num):
        """Generate a dataset with:
        * Sequence with the same IO token as io_names for sequences in IOI
        * Sequence with the same IO1 and IO2 token as io_names for sequences in ABC
        The template familiy (ABC/IOI) is preserved."""
        family_matching_templates = self.gen_templates(num, templates_to_match=templates, match_family=True)
        names = torch.tensor(self.rng.choice(self.all_names, (num, 3), replace=True))

        sequences_types = torch.tensor([1 if IO_TOK in template else 0 for template in templates])
        io_for_ioi = io_names * sequences_types + ordered_names[:, 0] * (
            1 - sequences_types
        )  # seq type is 1 for IOI, 0 for ABC. We keep the IO for IOI, the IO1 for ABC
        return self.gen_with_contraints(
            family_matching_templates, io_for_ioi, names[:, 1], ordered_names[:, 1], num
        )  # we give the IO2 for ABC as the third arguments (because we don't enforce order)

    def gen_for_template_and_names(self, templates, io_names, s_names, io2_names, num):
        """Generate a dataset with sequences with the same template as template, strongest form of matching"""
        matching_templates = self.gen_templates(num, templates_to_match=templates, match_position=True)

        return self.gen_with_contraints(matching_templates, io_names, s_names, io2_names, num)


# %%

# The code of IOICSDataset quite long but it's mostly a list of the attributes we'll need.
# Most of them are not directly useful but are nice to have around for debugging.
# We use attrs to define our attributes so there definition is propagated to the __init__ method
# without having to take care of that.

# `eq` says if the field shoul dbe used in equality tests while `init` says if the field should be used in the __init__ method.
# see [here](https://www.attrs.org/en/stable/api.html#attrs.field) for more details.

# the class is frozen so we can't directly set attributes after its motification. You can have a look at
# the section "Crash Course on the `attrs` library" from day 1 for more.

# %%


@attrs.frozen
class IOICSDataset(Dataset):  # will only deal with tokenized data

    tokens: torch.Tensor = attrs.field(init=True, eq=False)  # the inputs for the model
    labels: torch.Tensor = attrs.field(init=True, eq=False)
    # the labels of the IO classifcation task. [N1, N2] if ABC, [IO, EMPTY_LABEL] if IOI

    io_tokens: torch.Tensor = attrs.field(
        init=True, eq=False
    )  # on ABC, IO token is the first name of the sequence by convention
    s_tokens: torch.Tensor = attrs.field(init=True, eq=False)  # the subject
    io2_tokens: torch.Tensor = attrs.field(
        init=True, eq=False
    )  # on ABC, the second name of the first clause, on IOI, equals EMPTY_TOK

    templates: torch.Tensor = attrs.field(init=True, eq=False)  # the templates used to create each sequence
    sequences_types: torch.Tensor = attrs.field(init=True, eq=False)  # 1 if IOI, 0 if ABC
    string_sentences: list[str] = attrs.field(init=True, eq=False)
    names_in_order: torch.Tensor = attrs.field(init=True, eq=False)  # the names in the order of the sequence
    datasource: IOIDataSource = attrs.field(init=True, eq=False)

    N: int = attrs.field(init=True, eq=True)

    # input_names={"inputs"},
    # smarter way to set attributes for frozen class

    @classmethod  # class method so we can call it without instanciating the class. Main way to create the object
    def gen(cls, datasource: IOIDataSource, num):
        return cls(*datasource.gen(num))

    def __getitem__(self, idxs: rc.TorchAxisIndex):
        if isinstance(idxs, int):
            idxs = slice(idxs, idxs + 1)  # convert to slice so we have only one case to deal with

        new_toks = self.tokens[idxs]
        new_string_sentences = self.datasourremix_d5_part5_solutionw copy of the object with the given attributes modified
            self,
            arrs={name: rc.Array(inp.value[idxs], name) for name, inp in self.arrs.items()},
            tokens=self.tokens[idxs],
            labels=self.labels[idxs],
            io_tokens=self.io_tokens[idxs],
            s_tokens=self.s_tokens[idxs],
            io2_tokens=self.io2_tokens[idxs],
            templates=self.templates[idxs],
            sequences_types=self.sequences_types[idxs],
            string_sentences=new_string_sentences,
            names_in_order=self.names_in_order[idxs],
            N=new_toks.shape[0],
        )


#%% [markdown]
# Once we have this (overly) complicated apparaturs to generate our dataset,
#  we can finally define our own CondSampler!
# A `CondSampler` takes `ref` and `ds` as arguments and returns a new `Dataset`
# that matches `ref` in some way. Here we'll ignore the content `ds` and instead generate a new dataset using our
# fancy methods using its `IOIDataSource` object stored as an attribute.

# `N` is the size of the dataset and is always fixed.


class MatchingFirstNameSampler(CondSampler):
    def __call__(self, ref: Dataset, ds: Dataset, rng=None) -> IOICSDataset:
        assert isinstance(ref, IOICSDataset), type(ref)
        assert isinstance(ds, IOICSDataset), type(ds)
        args = ds.datasource.gen_for_first_name(ref.names_in_order, ds.N)
        return IOICSDataset(*args)


class MatchingSecondNameSampler(CondSampler):
    def __call__(self, ref: Dataset, ds: Dataset, rng=None) -> IOICSDataset:
        assert isinstance(ref, IOICSDataset), type(ref)
        assert isinstance(ds, IOICSDataset), type(ds)
        args = ds.datasource.gen_for_sec_name(ref.names_in_order, ds.N)
        return IOICSDataset(*args)


class MatchingThirdNameSampler(CondSampler):
    def __call__(self, ref: Dataset, ds: Dataset, rng=None) -> IOICSDataset:
        assert isinstance(ref, IOICSDataset), type(ref)
        assert isinstance(ds, IOICSDataset), type(ds)
        args = ds.datasource.gen_for_third_name(ref.names_in_order, ds.N)
        return IOICSDataset(*args)


class MatchingIOSampler(CondSampler):
    def __call__(self, ref: Dataset, ds: Dataset, rng=None) -> IOICSDataset:
        assert isinstance(ref, IOICSDataset), type(ref)
        assert isinstance(ds, IOICSDataset), type(ds)
        args = ds.datasource.gen_for_io(ref.io_tokens, ref.names_in_order, ref.templates, ds.N)

        return IOICSDataset(*args)


class MatchingIOAndPositionSampler(CondSampler):
    def __call__(self, ref: Dataset, ds: Dataset, rng=None) -> IOICSDataset:
        """If the sequence is ABC, then we match both IO1 and IO2. For sequences in IOI, we only match IO"""
        assert isinstance(ref, IOICSDataset), type(ref)
        assert isinstance(ds, IOICSDataset), type(ds)

        args = ds.datasource.gen_for_io_and_position(ref.io_tokens, ref.templates, ds.N)

        return IOICSDataset(*args)


class MatchingTemplateAndNameSampler(CondSampler):
    """The most restrictive sampler. The only variation allowed is in the name of the places and objects."""

    def __call__(self, ref: Dataset, ds: Dataset, rng=None) -> IOICSDataset:
        assert isinstance(ref, IOICSDataset), type(ref)
        assert isinstance(ds, IOICSDataset), type(ds)

        args = ds.datasource.gen_for_template_and_names(
            ref.templates, ref.io_tokens, ref.s_tokens, ref.io2_tokens, ds.N
        )
        return IOICSDataset(*args)


class MatchingFirstNameSamplerNoABCIOISwap(CondSampler):
    def __call__(self, ref: Dataset, ds: Dataset, rng=None) -> IOICSDataset:
        assert isinstance(ref, IOICSDataset), type(ref)
        assert isinstance(ds, IOICSDataset), type(ds)
        args = ds.datasource.gen_for_first_name(ref.names_in_order, ds.N, ABC_IOI_swap=False)
        return IOICSDataset(*args)


class MatchingSecondNameSamplerNoABCIOISwap(CondSampler):
    def __call__(self, ref: Dataset, ds: Dataset, rng=None) -> IOICSDataset:
        assert isinstance(ref, IOICSDataset), type(ref)
        assert isinstance(ds, IOICSDataset), type(ds)
        args = ds.datasource.gen_for_sec_name(ref.names_in_order, ds.N, ABC_IOI_swap=False)
        return IOICSDataset(*args)


class MatchingThirdNameSamplerNoABCIOISwap(CondSampler):
    def __call__(self, ref: Dataset, ds: Dataset, rng=None) -> IOICSDataset:
        assert isinstance(ref, IOICSDataset), type(ref)
        assert isinstance(ds, IOICSDataset), type(ds)
        args = ds.datasource.gen_for_third_name(ref.names_in_order, ds.N, ABC_IOI_swap=False)
        return IOICSDataset(*args)
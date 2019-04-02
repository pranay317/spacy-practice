#!/usr/bin/env python
# coding: utf8
"""Example of training an additional entity type

This script shows how to add a new entity type to an existing pre-trained NER
model. To keep the example short and simple, only four sentences are provided
as examples. In practice, you'll need many more — a few hundred would be a
good start. You will also likely need to mix in examples of other entity
types, which might be obtained by running the entity recognizer over unlabelled
sentences, and adding their annotations to the training set.

The actual training is performed by looping over the examples, and calling
`nlp.entity.update()`. The `update()` method steps through the words of the
input. At each word, it makes a prediction. It then consults the annotations
provided on the GoldParse instance, to see whether it was right. If it was
wrong, it adjusts its weights so that the correct action will score higher
next time.

After training your model, you can save it to a directory. We recommend
wrapping models as Python packages, for ease of deployment.

For more details, see the documentation:
* Training: https://spacy.io/usage/training
* NER: https://spacy.io/usage/linguistic-features#named-entities

Compatible with: spaCy v2.0.0+
Last tested with: v2.1.0
"""
from __future__ import unicode_literals, print_function

import plac
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding


# new entity label
AAPL_TICKER = "AAPL"
TRADE_BUY = "BUY"
TRADE_SELL = "SELL"
TRADE_QUANTITY = "SHARES_QUANTITY"

# training data
# Note: If you're using an existing model, make sure to mix in examples of
# other entity types that spaCy correctly recognized before. Otherwise, your
# model might learn the new type, but "forget" what it previously knew.
# https://explosion.ai/blog/pseudo-rehearsal-catastrophic-forgetting
TRAIN_DATA = [
    (
        "Apple Inc.",
        {"entities": [(0, 5, AAPL_TICKER)]},
    ),
    ("Apple Watch", {"entities": [(0, 5, AAPL_TICKER)]}),
    (
        "products from Apple are beautiful",
        {"entities": [(14, 19, AAPL_TICKER)]},
    ),
    ("The Apple event is going to happen in september",
     {"entities": [(4, 9, AAPL_TICKER)]}),
    ("I like to buy 100k shares of Apple", {"entities": [
     'O', 'O', 'O', 'U-'+TRADE_BUY, 'U-'+TRADE_QUANTITY, 'O', 'O', 'U-'+AAPL_TICKER]}),
    ("I like to sell 20k shares of Apple", {"entities": [
     'O', 'O', 'O', 'U-'+TRADE_SELL, 'U-'+TRADE_QUANTITY, 'O', 'O', 'U-'+AAPL_TICKER]}),
    ("I like to buy 3k shares of Apple", {"entities": [
     'O', 'O', 'O', 'U-'+TRADE_BUY, 'U-'+TRADE_QUANTITY, 'O', 'O', 'U-'+AAPL_TICKER]}),
    ("I like to buy 5k shares of Apple", {"entities": [
     'O', 'O', 'O', 'U-'+TRADE_BUY, 'U-'+TRADE_QUANTITY, 'O', 'O', 'U-'+AAPL_TICKER]}),
    ("I like to sell 35k shares of Apple", {"entities": [
     'O', 'O', 'O', 'U-'+TRADE_SELL, 'U-'+TRADE_QUANTITY, 'O', 'O', 'U-'+AAPL_TICKER]}),
    ("I like to sell 78k shares of Apple", {"entities": [
     'O', 'O', 'O', 'U-'+TRADE_SELL, 'U-'+TRADE_QUANTITY, 'O', 'O', 'U-'+AAPL_TICKER]}),
    ("I like to buy 129k shares of Apple", {"entities": [
     'O', 'O', 'O', 'U-'+TRADE_BUY, 'U-'+TRADE_QUANTITY, 'O', 'O', 'U-'+AAPL_TICKER]}),
    ("I like to buy 30k shares of Apple", {"entities": [
     'O', 'O', 'O', 'U-'+TRADE_BUY, 'U-'+TRADE_QUANTITY, 'O', 'O', 'U-'+AAPL_TICKER]}),
    ("I like to buy 15k shares of Apple", {"entities": [
     'O', 'O', 'O', 'U-'+TRADE_BUY, 'U-'+TRADE_QUANTITY, 'O', 'O', 'U-'+AAPL_TICKER]}),
    ("I like to sell 200k shares of Apple", {"entities": [
     'O', 'O', 'O', 'U-'+TRADE_SELL, 'U-'+TRADE_QUANTITY, 'O', 'O', 'U-'+AAPL_TICKER]}),
    ("I like to buy 330k shares of Apple", {"entities": [
     'O', 'O', 'O', 'U-'+TRADE_BUY, 'U-'+TRADE_QUANTITY, 'O', 'O', 'U-'+AAPL_TICKER]}),
    ("I like to buy 29k shares of Apple", {"entities": [
     'O', 'O', 'O', 'U-'+TRADE_BUY, 'U-'+TRADE_QUANTITY, 'O', 'O', 'U-'+AAPL_TICKER]}),
    ("I like to sell 2k shares of Apple", {"entities": [
     'O', 'O', 'O', 'U-'+TRADE_SELL, 'U-'+TRADE_QUANTITY, 'O', 'O', 'U-'+AAPL_TICKER]}),
    ("we'd recommend you sell apple. the price is almost at peak",
     {"entities": [(24, 29, AAPL_TICKER)]}),
    ("sell 50k of apple", {"entities": [
     (0, 4, TRADE_SELL), (5, 8, TRADE_QUANTITY), (12, 17, AAPL_TICKER)]}),
    ("Apple products are known to be expensive",
     {"entities": [(0, 5, AAPL_TICKER)]}),
    ("It is Apple's decision at the end of the day",
     {"entities": [(6, 11, AAPL_TICKER)]}),
    ("some one just bought 1m shares of a start up organisation", {
     "entities": [(14, 20, TRADE_BUY), (21, 23, TRADE_QUANTITY)]}),
    ("200k is quite a high quantity to buy right now", {
     "entities": [(0, 4, TRADE_QUANTITY), (33, 36, TRADE_BUY)]}),
    ("now is a best time to buy Apple shares a min of 20k shares", {'entities': [
        'O', 'O', 'O', 'O', 'O', 'O', 'U-'+TRADE_BUY, 'U-' + \
        AAPL_TICKER, 'O', 'O', 'O', 'O', 'U-'+TRADE_QUANTITY, 'O'
    ]})
]


@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    new_model_name=("New model name for model meta.", "option", "nm", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def main(model='../models/ticker', new_model_name="ticker", output_dir='../models/ticker', n_iter=40):
    """Set up the pipeline and entity recognizer, and train the new entity."""
    random.seed(0)
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")
    # Add entity recognizer to model if it's not in the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner)
    # otherwise, get it, so we can add labels to it
    else:
        ner = nlp.get_pipe("ner")

    ner.add_label(AAPL_TICKER)  # add new entity label to entity recognizer
    ner.add_label(TRADE_BUY)
    ner.add_label(TRADE_SELL)
    ner.add_label(TRADE_QUANTITY)
    # Adding extraneous labels shouldn't mess anything up
    ner.add_label("VEGETABLE")
    if model is None:
        optimizer = nlp.begin_training()
    else:
        optimizer = nlp.resume_training()
    move_names = list(ner.move_names)
    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        sizes = compounding(1.0, 10.0, 1.001)
        # batch up the examples using spaCy's minibatch
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            batches = minibatch(TRAIN_DATA, size=sizes)
            losses = {}
            # print('legth of batches', len(list(batches)))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer,
                           drop=0.35, losses=losses)
            print("Losses - "+str(itn+1)+"th iteration", losses)

    # test the trained model
    test_text = "take an order to sell 20k share of apple"
    doc = nlp(test_text)
    print("Entities in '%s'" % test_text)
    for ent in doc.ents:
        print(ent.label_, ent.text)

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta["name"] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        # Check the classes have loaded back consistently
        assert nlp2.get_pipe("ner").move_names == move_names
        doc2 = nlp2(test_text)
        for ent in doc2.ents:
            print(ent.label_, ent.text)


if __name__ == "__main__":
    plac.call(main)

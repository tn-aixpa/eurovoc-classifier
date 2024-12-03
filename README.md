# EuroVoc-Classifier

#### AIPC 
- ```kind```: product-template
- ```ai```: NLP
- ```domain```: PA

Multilingual text classifier based on Bert model and fine-tuned on official EU legal documents (from EUR Lex portal). This template
allows for creating a classifier for any EU language using the annotated EU datasets. The product contains operations for

- downloading (scraping) the data necessary for training using different types of labels and periods of time
- preprocessing the data in order to prepare it for training
- perform model training and registering the model
- serving the model as an optimized vLLM-based sequence classification API (Open Inference Protocol)
- serving the model using a custom API.

## Usage

Tool usage documentation [here](./docs/usage.md).

## How To

- [Download EUR Lex data](./docs/howto/download.md)
- [Preprocess data for training](./docs/howto/process.md)
- [Train the classifier model](./docs/howto/train.md)
- [Expose the classifed model as a service](./docs/howto/expose.md)
- [Evaluate](./docs/howto/evaluate.md)


## License

[Apache License 2.0](./LICENSE)

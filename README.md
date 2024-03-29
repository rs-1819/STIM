# Softprompt Training and Inference with MPT

This project demonstrates how to train and use softprompt embeddings with the MPT (Mosaic Pretrained Transformer) model for text generation tasks. The softprompt technique allows you to fine-tune the model's behavior by optimizing a small set of embeddings prepended to the input text.

## Requirements

- Python 3.x
- PyTorch
- Transformers library

## Installation

'''
TODO
'''

## Training Softprompt Embeddings

To train softprompt embeddings, use the `main.py` script. Modify the following variables according to your setup:

- `name`: The name of the MPT model to use (e.g., 'mosaicml/mpt-1b-redpajama-200b').
- `train_data`: A list of training sentences or a path to a file containing the training data.
- `batch_size`: The batch size for training.
- `num_epochs`: The number of training epochs.

Run the script to start training:

```
python main.py
```

The trained softprompt embeddings will be saved to a file named `softprompt_embeddings.pth`.

## Inference with Softprompt Embeddings

To perform inference using the trained softprompt embeddings, use the `inf.py` script. Modify the following variables:

- `name`: The name of the MPT model to use (same as in training).
- `softprompt_embedding_file`: The path to the trained softprompt embeddings file.

The script loads the model, tokenizer, and softprompt embeddings, and generates text based on a sample sentence.

Run the script to perform inference:

```
python inf.py
```

The generated text will be printed to the console.

## Customization

Feel free to customize the training data, model configuration, and inference parameters to suit your specific use case. You can also experiment with different softprompt sizes and training hyperparameters to achieve the desired performance.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- The MPT model is provided by MosaicML.
- The softprompt technique is based on the paper "The Power of Scale for Parameter-Efficient Prompt Tuning" by Brian Lester, Rami Al-Rfou, and Noah Constant.

If you have any questions or suggestions, please open an issue or submit a pull request.

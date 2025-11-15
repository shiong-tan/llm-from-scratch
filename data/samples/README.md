# Sample Training Data

This directory contains sample text data for training and testing the GPT model.

## Files

- `tiny-shakespeare.txt`: A small subset of Shakespeare's works for quick training experiments
- `simple-stories.txt`: Simple English sentences for basic language modeling

## Using Custom Data

To train on your own data:

1. **Prepare your text file**: Create a `.txt` file with your training data
2. **Tokenize**: Use the `Tokenizer` class to convert text to token IDs
3. **Create data loaders**: Use PyTorch's `DataLoader` for batching
4. **Train**: Pass to the `Trainer` class

See `examples/train_simple.py` for a complete example.

## Data Format

The training scripts expect plain text files. For best results:

- Use UTF-8 encoding
- Separate documents/examples with double newlines
- Remove excessive whitespace
- Include diverse examples

## Recommended Data Sources

For larger-scale training:

- [The Pile](https://pile.eleuther.ai/): 825GB of diverse text
- [OpenWebText](https://skylion007.github.io/OpenWebTextCorpus/): Reddit-filtered web text
- [Wikipedia dumps](https://dumps.wikimedia.org/): Clean encyclopedia text
- [Project Gutenberg](https://www.gutenberg.org/): Public domain books
- [Common Crawl](https://commoncrawl.org/): Web-scale data

## Data Preprocessing

For production training:

1. **Clean**: Remove HTML, special characters, excessive whitespace
2. **Filter**: Remove low-quality or inappropriate content
3. **Deduplicate**: Remove duplicate documents
4. **Tokenize**: Convert to token IDs with your tokenizer
5. **Shuffle**: Randomize order for better training

## Privacy and Legal

- Only use data you have rights to use
- Respect copyright and licensing
- Remove personally identifiable information (PII)
- Follow data usage agreements

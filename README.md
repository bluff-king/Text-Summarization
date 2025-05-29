# Text Summarization

This repository contains the code and report for a project on English Text Summarization Using Deep Learning Methods.

## Structure

The project is organized as follows:

- `Abstractive/`: Contains Jupyter notebooks for different abstractive summarization models, including custom Transformer, T5-small, BART-base, Pegasus-custom, and an Ensemble model.
- `dataft/`: Data used for fine-tuning the models.
- `dataset/`: The raw or processed dataset.
- `Embedding/`: GloVe embeddings file
- `Model/`: Saved model checkpoints.
- `demo.py`: Script for demonstrating the summarization models with the help of Gradio
- `eda&preprocess.ipynb`: Jupyter notebook for exploratory data analysis and preprocessing.

## Data

The project utilizes a preprocessed version of the [CNN/DailyMail dataset](https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail) from Kaggle. The dataset consists of news articles and their corresponding abstractive summaries.

## Introduction and Demo

This project focuses on abstractive text summarization using deep learning models. The goal is to generate concise summaries that retain the core content of the original text. The project explores various Transformer-based models, including fine-tuned pre-trained models like T5-small, BART-base, and Pegasus, as well as a custom Transformer implementation.

A demo of the text summarization models is available.

[Image of demo will be added here later]

## Requirements

The project requires Python and several libraries for deep learning, natural language processing, and data manipulation. Specific requirements can be inferred from the Jupyter notebooks and potentially a `requirements.txt` file if present (not listed in the provided file structure, but common in such projects). Key libraries likely include:

- PyTorch 
- Transformers (Hugging Face library)
- NLTK
- pandas
- numpy

## Acknowledgements

This project was conducted as part of the Project II course at Hanoi University of Science and Technology, supervised by Ths. Le Duc Trung.

The project utilizes concepts and data from the following sources:

[1] S. Banerjee and A. Lavie. Meteor: An automatic metric for mt evaluation with improved cor-
relation with human judgments. In Proceedings of the ACL Workshop on Intrinsic and Extrinsic
Evaluation Measures for MT and/or Summarization, pages 65–72, 2005.
[2] P. Gowrishankar. Newspaper text summarization - cnn/dailymail. https://www.
kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail,
2023. Accessed from Kaggle: https://www.kaggle.com/datasets/gowrishankarp/
newspaper-text-summarization-cnn-dailymail.
[3] M. Kirmani, G. Kaur, and M. Mohd. Analysis of abstractive and extractive summarization meth-
ods. International Journal of Emerging Technologies in Learning (iJET), 19(01):86–96, January
2024. doi: 10.3991/ijet.v19i01.46079. URL https://doi.org/10.3991/ijet.v19i01.46079.
[4] M. Lewis, Y. Liu, N. Goyal, M. Ghazvininejad, A. Mohamed, O. Levy, V. Stoyanov, and L. Zettle-
moyer. Bart: Denoising sequence-to-sequence pre-training for natural language generation,
translation, and comprehension. arXiv preprint arXiv:1910.13461, 2019. doi: 10.48550/arXiv.
1910.13461. URL https://arxiv.org/abs/1910.13461.
[5] C.-Y. Lin. Rouge: A package for automatic evaluation of summaries. In Text Summarization
Branches Out: Proceedings of the ACL-04 Workshop, pages 74–81, 2004.
[6] H. Nguyen, H. Chen, L. Pobbathi, and J. Ding. A comparative study of quality evaluation
methods for text summarization, 2024. URL https://arxiv.org/abs/2407.00747. Submitted
to EMNLP 2024.
[7] K. Nguyen. N-gram language models. part 1: The unigram model, 2020. URL https://medium.
com/mti-technology/n-gram-language-model-b7c2fc322799. Accessed: 2025-05-16.
[8] J. Pennington, R. Socher, and C. D. Manning. Glove: Global vectors for word representation
(6b, 50d). https://www.kaggle.com/datasets/watts2/glove6b50dtxt, 2014. Pretrained on
Wikipedia 2014 + Gigaword 5, 6B tokens, 400K vocab, 50d vectors. Accessed: 2025-05-14.
[9] G. PIAO. What is bpe: Byte-pair encoding?, 2025. URL https://medium.com/@parklize/
what-is-bpe-byte-pair-encoding-5f1ea76ea01f. Accessed: 2025-05-16.
[10] C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu.
Exploring the limits of transfer learning with a unified text-to-text transformer. arXiv preprint
arXiv:1910.10683, 2020. doi: 10.48550/arXiv.1910.10683. URL https://arxiv.org/abs/
1910.10683.
[11] X. Song and D. Zhou. A fast wordpiece tokenization system, 2021. URL https://research.
google/blog/a-fast-wordpiece-tokenization-system/. Accessed: 2025-05-16.
[12] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and
I. Polosukhin. Attention is all you need. arXiv preprint arXiv:1706.03762, 2017. doi:
10.48550/arXiv.1706.03762. URL https://arxiv.org/abs/1706.03762.
[13] J. Zhang, Y. Zhao, M. Saleh, and P. J. Liu. Pegasus: Pre-training with extracted gap-sentences
for abstractive summarization. arXiv preprint arXiv:1912.08777, 2020. doi: 10.48550/arXiv.
1912.08777. URL https://arxiv.org/abs/1912.08777.
[14] T. Zhang, V. Kishore, F. Wu, K. Q. Weinberger, and Y. Artzi. Bertscore: Evaluating text genera-
tion with bert. In International Conference on Learning Representations (ICLR), 2020.

## License

This project is licensed under the [License Name] - see the [LICENSE](LICENSE) file for details.

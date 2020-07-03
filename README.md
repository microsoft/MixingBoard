# MixingBoard: a Knowledgeable Stylized Integrated Text Generation Platform

# News

# Setup

## Package installation
## API accounts
### Bing Search
### Text-to-Speech

# Modules

## Knowledge passage retrieval
We use the following unstructured free-text sources to retrieve relevant knowledge passage: search engine, specialized websites (e.g. wikipedia), and user provided document.
```
>>> python src/knowledge.py
>>> QUERY:  what is deep learning?
```
E.g. the above command calls Bing search API and returns the following results
```
URL:    https://en.wikipedia.org/wiki/Deep_learning
TXT:    Deep learning is a class of machine learning algorithms that (pp199–200) uses multiple layers to progressively extract higher level features from the raw input. For example, in image processing, lower layers may identify edges, while higher layers may identify the concepts relevant to a human such as digits or letters or faces.. Overview. Most modern deep learning models are based on ...

URL:    https://machinelearningmastery.com/what-is-deep-learning/
TXT:    Deep Learning is Large Neural Networks. Andrew Ng from Coursera and Chief Scientist at Baidu Research formally founded Google Brain that eventually resulted in the productization of deep learning technologies across a large number of Google services.. He has spoken and written a lot about what deep learning is and is a good place to start. In early talks on deep learning, Andrew described deep ...

URL:    https://www.forbes.com/sites/bernardmarr/2018/10/01/what-is-deep-learning-ai-a-simple-guide-with-8-practical-examples/
TXT:    Since deep-learning algorithms require a ton of data to learn from, this increase in data creation is one reason that deep learning capabilities have grown in recent years.
```

## Open-ended dialogue generation
We use [DialoGPT](https://github.com/microsoft/DialoGPT) as an example.
```
>>> python src/open_dialog.py
CONTEXT:        What's your dream?
```
E.g. the above command calls DialoGPT model and returns the following results
```
DPT 1.008       First one is to be a professional footballer. Second one is to be a dad. Third one is to be a firefighter.
DPT 1.007       First one is to be a professional footballer. Second one is to be a dad. Third one is to be a father of two.
...
```

## Machine reading comprehension
```
>>> python src/mrc.py
QUERY:          Who is Jeffrey Hinton?
PASSAGE:        Geoffrey Everest Hinton CC FRS FRSC is an English Canadian cognitive psychologist and computer scientist, most noted for his work on artificial neural networks. Since 2013 he divides his time working for Google and the University of Toronto. In 2017, he cofounded and became the Chief Scientific Advisor of the Vector Institute in Toronto.
```
E.g. the above command calls [BiDAF](https://allenai.github.io/bi-att-flow/) model and returns the following results
```
Bidaf 0.352     an English Canadian cognitive psychologist and computer scientist
```

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

# Disclaimer

# Citation

If you use this code in your work, you can cite our [arxiv](https://arxiv.org/abs/2005.08365) paper:

```
@article{gao2020mixingboard,
  title={MixingBoard: a Knowledgeable Stylized Integrated Text Generation Platform},
  author={Gao, Xiang and Galley, Michel and Dolan, Bill},
  journal={Proc. of ACL},
  year={2020}
}
```
##  Research Motivation & Overview

The economic consequences of political uncertainty have long been a focal point in political economy and financial economics. While classical models like the Efficient Market Hypothesis (EMH) suggest that prices fully reflect public information, newer theories highlight heterogeneity in investor behavior, attention allocation, and narrative amplification.

This project investigates whether media discourse and market behavior demonstrate synchronized sentiment cycles during episodes of heightened uncertainty, particularly in relation to tariff policy shocks.

We hypothesize that emotionally charged, policy-specific media narratives—such as coverage of tariff escalations—not only reflect macroeconomic expectations but actively shape them. Specifically, we ask:  
**Do tariff-related sentiment signals trigger more directional market responses than general financial sentiment?**

To address this, we design a comparative framework that isolates the causal influence of topic-specific sentiment by curating two parallel datasets:

- **Treatment group**: Wall Street Journal articles filtered for “tariff”, “trade war”, “import/export”, and related terms.
- **Control group**: General financial news articles from Bloomberg over the same time frame.

By controlling for time window and model architecture, and varying only the thematic scope of media input, we assess whether markets are more reactive to targeted uncertainty narratives than to ambient sentiment.

##  Scalable Architecture & Methodology

Due to the scale and complexity of the problem, this project employs a multi-stage pipeline that integrates scalable cloud computing infrastructure with deep learning methods for text and time series modeling.

### Why scalable methods are necessary:

- The volume of financial news is extremely large. Scraping platforms like Bloomberg and the Wall Street Journal involves navigating bot-detection systems and dynamic JavaScript-rendered content, which cannot be handled efficiently with simple scripts. We adopt a modular, stepwise crawler design using Playwright and Selenium on AWS EC2 to mimic human browsing behavior.
  
- Sentiment analysis is computationally intensive, especially when applied to thousands of daily records. To ensure performance and scalability, we deploy FinBERT— a transformer-based financial sentiment classifier—via Amazon SageMaker’s **Batch Transform** service, enabling parallel sentiment inference on large batches of text streamed from Amazon S3.

- To extend the scope of our study beyond current events and improve robustness, we integrate the **FNSPID dataset**, which contains over **15.7 million financial news articles** and **29.7 million stock price records** from 2019–2020. This allows us to perform historical backtesting on prior tariff shocks.

- Finally, forecasting is done using a deep learning model—**LSTM (Long Short-Term Memory)**—implemented in PyTorch and trained on the University of Chicago’s Midway High-Performance Computing (HPC) cluster. This architecture enables efficient modeling of sequential dependencies in financial time series data with high-dimensional inputs including sentiment and technical indicators.

### Summary of scalable computing components:
### Summary of Scalable Computing Components (by Subgroup)

| Task                  | Bloomberg + Midway (Baihui Wang)                                 | Wall Street Journal + AWS (Charlotte Li)                            |
|-----------------------|--------------------------------------------------------------------|-------------------------------------------------------------------------|
| Web scraping          | Playwright on EC2 (simulates human mouse behavior to bypass bot detection) | Selenium + Requests on EC2 (multi-stage crawler targeting WSJ tariff articles) |
| Sentiment analysis    | FinBERT on Midway (Hugging Face Transformers, local inference)     | FinBERT deployed on SageMaker (Batch Transform using S3 CSV inputs) + matched S&P500 stock data (open, high, low, close, volume) |
| Historical data       | FNSPID dataset: 15.7M news + 29.7M prices (2019–2020 full market span) | Tariff-topic WSJ articles: ~2000 in 2019 and ~1000 in 2020              |
| Sequential modeling   | PyTorch LSTM trained on SageMaker and locally                     | PyTorch LSTM trained on SageMaker and locally                          |
| Backtesting           | Bloomberg-based sentiment models tested on 2019–2020 tariff windows | WSJ-based models backtested on tariff shocks in 2019 and 2020           |

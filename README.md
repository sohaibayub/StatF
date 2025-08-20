# StatF

StatF is a research-focused project that explores the relationship between static website features and their energy consumption. The project utilizes data collected from Lighthouse audits and WebPageTest results to analyze how various frontend characteristics impact the energy efficiency of websites.

## 📊 Project Overview

The primary objective of StatF is to provide insights into how different frontend features—such as performance scores, resource usage, and architectural patterns—correlate with the energy consumption of static websites. This can aid developers and researchers in optimizing web applications for better energy efficiency.

## 🔧 Features

- **Data Collection**: Integration with Lighthouse and WebPageTest to gather performance and resource usage metrics.
- **Data Analysis**: Scripts to analyze and visualize the relationship between frontend features and energy consumption.
- **Feature Correlation**: Tools to identify and quantify correlations between various web metrics and energy usage.

## 🗂 Project Structure

```plaintext
StatF/
├── .gitignore
├── DetailsDataColumnsList.txt
├── LighthouseDataColumnsList.txt
├── SummaryDataColumnsList.txt
├── FeatureBasedDistributionOfPower.py
├── LHperfScrAndPow.py
├── correlationOfFeaturesWithPower.py
└── README.md

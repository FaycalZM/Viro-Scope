# ViroScope - Global Disease Outbreak Monitoring System 🧑‍⚕️📈
## Overview
ViroScope is a cutting-edge, real-time data pipeline designed to track and analyze global health data, providing early warnings of potential disease outbreaks. By leveraging public health-related data, AI-powered data processing, and [**Fluvio**](https://www.fluvio.io/)'s real-time data streaming capabilities, this system offers a comprehensive solution for monitoring, detecting, and responding to emerging health threats on a global scale.
[![Quira Vote](https://img.shields.io/badge/Quira-View%20Repo-blue)](quira-vote-link-here)
## Table of Contents 📑
- [Features](#features)
- [Technology Stack](#technology_stack)
- [YouTube Demonstration](#youtube-demonstration)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Contribution](#contribution)
- [Support](#support)
## Features 🌟
- Real-time data processing pipeline using Fluvio CLI and FLuvio' python SDK, and Time Series Forecasting using LSTM model 
- Predictions analysis & Anomaly detection: identify potential health outbreaks by understanding future trends of disease-related searches
- Model evaluation & visualization
## Technology Stack ⚙️
- python with flask framework
- fluvio CLI & fluvio python SDK 
- pandas, numpy for data preprocessing
- scikit-learn, keras for model creation and training
- matplotlib for visualization
## YouTube Demonstration 📹
[Watch the video](https://www.youtube.com/watch?v=w4fCBKc0mjA)
## Requirements 🧰
- Python 3.7 or higher
- Rust compiler (for fluvio) 
## Installation ⏳
1. **Clone the repository**:
    ```bash
    git clone https://github.com/FaycalZM/Viro-Scope.git
    cd Viro-Scope
    ```

2. **Create virtual env & activate it & Install required packages**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3. **Set up Fluvio**:
    -  run the *flv_setup.sh* in the scripts directory:
	    ```bash
        ./scripts/flv_setup.sh
        ```

## Usage 🌎
1. Run the *pipeline.sh* script in the scripts directory to start the pipeline execution, and wait for magic to happen:
    ```bash
    ./scripts/pipeline.sh  
    ```

2. Data is fetched every 24 hours, but you can trigger the pipeline execution manually by sending a GET request to http://localhost:5000/fetch-trends :
	 ```bash
	 curl http://localhost:5000/fetch-trends
    ```

3. Run the *model_evaluation.py* script separately for additional informations:
    ```bash
    python src/model_evaluation.py
    ```
## Contributing 🏗️
1. **Fork the Repository:**
    - Create a fork of the repository.
2. **Open a Pull Request:**
    - Open a pull request from your forked repository to the main repository.
## Support 👍
If you like this project, please support it by upvoting on Quira and starring the GitHub repository!
[![Quira Repo](https://img.shields.io/badge/Quira-View%20Repo-blue)](quira-vote-link-here)
Thank you for your support!
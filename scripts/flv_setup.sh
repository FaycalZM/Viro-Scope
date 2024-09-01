#!/bin/bash

# exit immediately if a command exits with a non-zero status
set -e

# list of topics to create
TOPICS=(
    "google-trends-data"
    "preprocessed-trends-data"
    "predictions-data"
    "anomalies-data"
)

# topics configuration goes here
# retention-time = 1 day means the topic data will be deleted after 1 day
declare -A TOPICS_CONFIG
TOPICS_CONFIG["google-trends-data"]="--retention-time 1d"
TOPICS_CONFIG["preprocessed-trends-data"]="--retention-time 1d"
TOPICS_CONFIG["predictions-data"]="--retention-time 1d"
TOPICS_CONFIG["anomalies-data"]="--retention-time 1d"


install_fluvio_cli() {
    echo "Checking for Fluvio CLI installation..."
    if ! command -v fluvio &> /dev/null; then
        echo "Fluvio not found. Installing Fluvio CLI..."
        curl -fsS https://hub.infinyon.cloud/install/install.sh | bash
        # add fluvio CLI to PATH
        echo 'export PATH="${HOME}/.fvm/bin:${HOME}/.fluvio/bin:${PATH}"' >> ~/.bashrc
        echo 'source "${HOME}/.fvm/env"' >> ~/.bashrc
        # Source the profile to update PATH for the current session
        source ~/.profile || source ~/.bashrc || source ~/.zshrc
        echo "Fluvio CLI installed successfully."
    else
        echo "FLuvio CLI already installed."
    fi
}

start_fluvio_cluster(){
    echo "Starting Fluvio cluster..."
    fluvio cluster start
    echo "Fluvio cluster started successfully."
    echo "Checking cluster status..."
    fluvio cluster status
}


create_fluvio_topic(){
    local topic=$1
    local config=${TOPICS_CONFIG[$topic]}

    echo "Creating topic: $topic..."
    if fluvio topic list | grep -q "^$topic$"; then
        echo "Topic '$topic' already exists. Skipping creation."
    else
        if [ -z "$config" ]; then
            fluvio topic create "$topic"
        else
            fluvio topic create "$topic" $config
        fi
    fi
}

create_all_topics() {
    echo "Creating all topics..."
    for topic in "${TOPICS[@]}"; do
        create_fluvio_topic "$topic"
    done
    echo "All topics created successfully."
}

list_all_topics(){ 
    echo "Listing all FLuvio topics..."
    fluvio topic list
}

main() {
    install_fluvio_cli
    start_fluvio_cluster
    create_all_topics
    list_all_topics
    echo "FLuvio setup completed successfully."
}


main


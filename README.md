graph TD
    %% Define Styles
    classDef default fill:#f9f9f9,stroke:#333,stroke-width:2px;
    classDef startEnd fill:#d4edda,stroke:#28a745,stroke-width:2px;
    classDef linkNode fill:#e2f0d9,stroke:#385723,stroke-width:1px;

    %% Nodes
    Start([Week 1/2 Kickoff]) :::startEnd
    
    subgraph EDA [1. Exploratory Data Analysis]
        Spider[Spider Dataset]
        Bird[BIRD Bench Dataset]
        HF[Hugging Face Viewer]
    end
    
    Pick[2. Pick Suitable Dataset]
    Convert[3. Convert SQLite DB to JSON]
    
    subgraph LLM [4. Model Prompting & Inference]
        SelectSQL[Select Diverse SQL Complexity Levels]
        PromptQwen[Prompt Qwen 2.5 Coder]
        PromptBase[Prompt Baselines: GPT-5 / Sonnet 4.6](baseline_model_prompt.md)
    end
    
    subgraph Evaluation [5. Comparison & Analytics]
        Metrics[Compute Metrics: BERTScore / BLEU]
        Visuals[Generate Visualization Reports]
    end
    
    End([Week 2 Deliverable Ready]) :::startEnd

    %% Links/Flow
    Start --> Spider
    Start --> Bird
    Start --> HF
    
    Spider --> Pick
    Bird --> Pick
    HF --> Pick
    
    Pick --> Convert
    Convert --> SelectSQL
    
    SelectSQL --> PromptQwen
    SelectSQL --> PromptBase
    
    PromptQwen --> Metrics
    PromptBase --> Metrics
    
    Metrics --> Visuals
    Visuals --> End

    %% Hyperlinks for Interactive Rendering
    click Spider "https://yale-lily.github.io/spider" "Go to Spider Project"
    click Bird "https://bird-bench.github.io/" "Go to BIRD Bench"
    click HF "https://huggingface.co/datasets/xlangai/spider/viewer/spider" "Open HF Viewer"
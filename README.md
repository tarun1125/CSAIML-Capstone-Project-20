```mermaid
graph TD
    %% Define Styles
    classDef default fill:#f9f9f9,stroke:#333,stroke-width:2px;
    classDef startEnd fill:#d4edda,stroke:#28a745,stroke-width:2px;

    %% Nodes
    Start([Week 1/2 Kickoff])
    
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
        PromptBase[Prompt Baselines: GPT-5 / Sonnet 4.6]
    end
    
    subgraph Evaluation [5. Comparison & Analytics]
        Metrics[Compute Metrics: BERTScore / BLEU]
        Visuals[Generate Visualization Reports]
    end
    
    End([Week 2 Deliverable Ready])

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

    %% Apply Styles Explicitly (Fixes the Parse Error)
    class Start startEnd;
    class End startEnd;

    %% Hyperlinks & Local Repo Files
    click Spider "https://github.io" "Go to Spider Project"
    click Bird "https://github.io" "Go to BIRD Bench"
    click HF "https://huggingface.co" "Open HF Viewer"
    click PromptBase "./baseline_model_prompt.md" "Open Baseline Model Prompt"
```



## GPT Execution Accuracy : 12/12
## QWEN 2.5 coder Execution Accuracy : 8/12

## BLEU Scores
     Model  codebleu  ngram_match_score  weighted_ngram_match_score  \
0  ChatGPT    0.5812             0.2504                      0.2659   
1  Qwen2.5    0.4134             0.0307                      0.0399   

   syntax_match_score  dataflow_match_score  
0              0.8085                     0  
1              0.5831                     0  

## BERT SCORE
Average Scores
         precision  recall      f1
model                             
ChatGPT     0.8401  0.7337  0.7853
Qwen2.5     0.5868  0.3473  0.4630

## At a Glance
| Metric             |    ChatGPT |       Qwen |
| ------------------ | ---------: | ---------: |
| Execution Accuracy |      91.7% |      58.3% |
| CodeBLEU           | **0.5812** | **0.4134** |
| BERTScore F1       | **0.7853** | **0.4630** |

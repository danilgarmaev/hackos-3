# Welcome to HackOS 3: DeepSeek!
This is the GitHub repository containing our dataset(s) and benchmarking tools. 
Big shoutout to Rootly for providing their dataset for this hackathon; their repository containing the full dataset can be found here:
[system-logs-dataset by Rootly](https://github.com/rootlyhq/system-logs-dataset)


## About the Data
The datasets provided by Rootly are sourced from production-environment system logs. 

We have two datasets: [**synthetic**](https://github.com/aniketsrinivasan/hackos-3/tree/main/data/synthetic) and [**actual**](https://github.com/aniketsrinivasan/hackos-3/tree/main/data/actual) data. 
Both datasets contain system log information. The actual dataset was provided by Rootly. The synthetic dataset was fully generated using **DeepSeek-R1:70b**. 

Each dataset contains the following metrics:

* `input`: the line from the system logs.
* `error_type`: the type of error that the line represents, one of "fatal", "runtime", "warning" or "no_error".
* `severity`: the severity (LogLevel) of the error, one of "error", "warn", or "notice".
* `description`: a brief interpretation of the log line (generated using DeepSeek-R1:70b).
* `solution`: a brief list of possible solutions to the error, if any (generated using DeepSeek-R1:70b).

## About the Task
Our main task is to improve models' ability and speed to perform tasks on the dataset (in particular, we want to be able to improve their accuracy in 
reproducing DeepSeek-R1's `description` and `solution` metrics). 


## Benchmarking
To make benchmarking models more straightforward, we've implemented a `Benchmark` class (in `benchmark/benchmark.py`), and a `Model` class (in `benchmark/model_class.py`). 
If you want to benchmark your models on the datasets here, feel free to use our existing pipeline to do so. 

There's also an example model in `example_model.py` for how you can set up models to work with the pipeline.

If you have any question, feel free to reach out or ping Aniket (@anixus) or Laurence (@larryl4643) on Discord!

Think about the following metrics while benchmarking:
* How fast does your model run?
* How small is your model?
* How accurate is your model?

For comparison, **we ran GPT-4o on a small portion of the dataset**, and it performs poorly on almost all metrics (~40% accuracy) on the actual data. 



Once again, thank you to Rootly for providing us with our data! 

![Rootly Logo](https://github.com/aniketsrinivasan/hackos-3/blob/main/rootly.com-logo.png)

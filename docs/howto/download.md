# How to download EUR Lex data

To obtain the necessary training data, it is possible to use ``download`` operation that performs data scraping. 

1. Initialize the project

```python
import digitalhub as dh
PROJECT_NAME = "eurovoc-classifier-project" # here goes the project name that you are creating on the platform
proj = dh.get_or_create_project(PROJECT_NAME)
```

2. Define the function

Register the ``download`` function in the project

```python
func = project.new_function(name="download",
                                kind="python",
                                python_version="PYTHON3_10",
                                code_src="git+https://github.com/tn-aixpa/eurovoc-classifier",
                                handler="src.download:download",
                                requirements=["tqdm==4.64.1", "requests==2.28.1", "beautifulsoup4==4.11.2", "lxml==4.9.2" ,"languagecodes==1.1.1", "PageRange==0.4"])
```
The function represent a Python operation and may be invoked directly locally or on the cluster.

3. Run the download function

```python
func.run(action="job", parameters={"lang": "it", "year": "2022", "label_types": "TC,DO,MT"})
```

Here the data for Italian language is used, considering only year 2022 and all the label types. See [Usage](../usage.md) for further details.

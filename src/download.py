import json
from os import listdir, path, makedirs
import gzip
from tqdm import tqdm
import argparse
from src.scraper import EURlexScraper

base_path = "./download"
output_path = "./download/output"

def extract_documents(data_path, output_path):
    path_docs = data_path
    years = [name for name in listdir(path_docs) if path.isfile(path.join(path_docs, name)) and name.endswith(".json.gz")]
    final_path = output_path

    makedirs(final_path, exist_ok=True)

    print(f"Working on data from {path_docs}")
    
    for year in tqdm(years):
        with gzip.open(path.join(path_docs, year), "rt", encoding="utf-8") as f:
            data = json.load(f)
            to_del = set()
            for doc in data:
                # For each document in the file, only keep those with at least one Eurovoc classifier and without an empty text
                if len(data[doc]["eurovoc_classifiers"]) == 0 or data[doc]["full_text"] == "":
                    to_del.add(doc)
            for doc in to_del:
                del data[doc]
            with gzip.open(path.join(final_path, year), "wt", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)

def download(
    project,
    lang="it",
    year="",
    category="",
    label_types="TC",
    max_retries=10,
    sleep_time=1
):

    directory = f"{base_path}/eurlexdata/input/"
    output_directory = f"{base_path}/eurlexdata/output/"
    
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--language", type=str, default="it", help="Language to scrape.")
    parser.add_argument("--year", type=str, default="", help="Years to scrape.")
    parser.add_argument("--category", type=str, default="", help="Categories to scrape.")
    parser.add_argument("--label_types", type=str, default="TC", help="Label types to scrape. Use comma separated values for multiple types. Accepted values: TC (Thesaurus Concept), MT (Micro Thesaurus), DO (Domain).")
    parser.add_argument("--max_retries", type=int, default=10, help="Maximum number of retries for each page, both search pages and individual documents.")
    parser.add_argument("--sleep_time", type=int, default=1, help="Sleep time between document requests.")
    parser.add_argument("--directory", type=str, default="./eurlexdata/", help="Directory for the saved data.")
    
    args = [
        "--language", lang,
        "--year", year,
        "--category", category,
        "--label_types", label_types,
        "--max_retries", str(max_retries),
        "--sleep_time", str(sleep_time),
        "--directory", directory
    ]
    args = parser.parse_args(args)

    scraper = EURlexScraper(lang=args.language, log_level=2)

    if args.year == "":
        if args.category == "":
            documents = scraper.get_documents_by_year(
                years=[],
                save_data=True,
                save_html=False,
                directory=args.directory,
                resume=False,
                max_retries=args.max_retries,
                sleep_time=args.sleep_time,
                skip_existing=True,
                label_types=args.label_types,
            )
        else:
            documents = scraper.get_documents_by_category(
                categories=args.category.split(","),
                save_data=True,
                save_html=False,
                directory=args.directory,
                resume=False,
                max_retries=args.max_retries,
                sleep_time=args.sleep_time,
                skip_existing=True,
                label_types=args.label_types,
            )
    else:
        if args.category != "":
            raise("You can't specify both a category and a year.")
        documents = scraper.get_documents_by_year(
            years=args.year,
            save_data=True,
            save_html=False,
            directory=args.directory,
            resume=False,
            max_retries=args.max_retries,
            sleep_time=args.sleep_time,
            skip_existing=True,
            label_types=args.label_types,
        )
    extract_documents(f"{directory}/{lang}", f"{output_directory}/{lang}")
    if project:
        project.log_artifact(name="classified_data", kind="artifact", source=output_directory)
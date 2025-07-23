from langchain.text_splitter import CharacterTextSplitter

text = """
Enumerating objects: 8, done.
Counting objects: 100% (8/8), done.
Delta compression using up to 12 threads
Compressing objects: 100% (7/7), done.
Writing objects: 100% (7/7), 1.43 KiB | 244.00 KiB/s, done.
Total 7 (delta 1), reused 0 (delta 0), pack-reused 0 (from 0)
remote: Resolving deltas: 100% (1/1), completed with 1 local object.
To https://github.com/justxperimenting/Langchain.git
   2683308..b1dcbba  main -> main
"""

splitter = CharacterTextSplitter(
    chunk_size = 60,
    chunk_overlap = 0,
    separator=""
)

result = splitter.split_text(text)

print(result)

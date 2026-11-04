# last-rag

so the whole idea is, as we move to the next level of the rag, we try each of the ways, the known cheap ways and the best possible ways, one can
implement in the RAG architecture and then we check everything setting a benchmark for the new people and yeah like said, no method is better,
its all about the use-case and stuff..

## Disclaimer

This benchmarking is not a substitute for your own judgement. The results are not always indicative of the best approach for a given use case.
there are few points to note:

1. The dataset is WixQ QnAs. This may or may not be representative of your use case.
2. The results are not always indicative of the best approach for a given use case. You should always use your own judgement when selecting a RAG component, and better to look at your use-cases as well.
3. No LangChain, LangGraph or any third-party libraries were used in this benchmarking. This is solely a model based approach. The things i have used in making and using are referenced [here](<>).
4. I aint no Linus torvalds, dont expect me to write the best code, I am learning and improving. This is only a benchmarking experiment.
5. Any improvements or suggestions are welcome.

## WHAT? - I just want to :)

last-rag is an educational, mesaured RAG system. It uses open source WixQ QnAs as a knowledge base. As the name suggests, it benchmarks every single possible RAG component against the queries it receives. The final product may look like a toggle metrics, where users can switch between different RAG components and see how they perform. This gave them the idea of whether or not how each of the RAG components performed against the queries they received.

> The point to make this is to show that using all the RAG components together is not always the best. It is often better to use a simpler, more focused approach that focuses on a single RAG component. Based on the results of this benchmarking, it is clear that using all the RAG components together is not always the best approach. Also, it is important to note that the results of this benchmarking are not always indicative of the best approach for a given use case. You should always use your own judgement when selecting a RAG component, and better to look at your use-cases as well.

## WHY? - A reference to the Past

The idea of this experiment came from my previous internship where i was working on the RAG pipeline and specifically the retrieval pipeline. I tried several nuanced apprach and naive approaches but the results were not commendable. I ultimately used a weighted approach to improve the results. But this was a hard and static way to improve the results. I wanted to see if i could do better, eventually i got to know about the re-ranking and cross-encoders for the top-k retrieval. I researched a lot and came up with again a dumb approach (i was learning) though it only improved the results marginally. I changed the models, like instead of using dense retrieval models, i used sparse retrieval models. I also tried swapping the embedding models, but again nothing much, eventually, i continued with the project and it got handled by a senior dev. I was then working on the api-designing, integrations and vectorDB optimisations.
After an year, I thought i should give it a try to make it better. Actually, i watched a video on RAG improvements and i was inspired to try out the re-ranking and cross-encoders. But thought what were the odds of adding other components, and just to check whether it would improve the results. With a lot of things in the market now, like people are using decision models to bifurcate the ambigous, meta queries, with the one that are genuine to the knowledge base and they are getting good results. Hence, I decided to just add all the things and make it interactive using toggle buttons to let users see how the results change with different configurations.

## HOW? - the benchmarkmaxxing

So i had a chat with my companion (claude) and came with this idea of toggle and let users see how the results change with different configurations. I have been desiging the structure, but first, i am sourcing the tools and ofcourse the data for the benchmarking process. I am writing standalone scripts with the tools and collecting the results as an in-house testing.

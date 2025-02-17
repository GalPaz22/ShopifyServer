import express from "express";
import bodyParser from "body-parser";
import { MongoClient, ObjectId } from "mongodb";
import { OpenAI } from "openai";
import cors from "cors";
import dotenv from "dotenv";
import Anthropic from '@anthropic-ai/sdk';
import { GoogleGenerativeAI } from "@google/generative-ai";


dotenv.config();

const app = express();
app.use(bodyParser.json());
app.use(cors({ origin: "*" }));


// Initialize OpenAI client

const genAI = new GoogleGenerativeAI( process.env.GOOGLE_API_KEY );
const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash-001" });
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const mongodbUri = process.env.MONGODB_URI;

// Create a persistent MongoDB client
let client;

async function connectToMongoDB(mongodbUri) {
  if (!client) {
    client = new MongoClient(mongodbUri, {
      useNewUrlParser: true,
      useUnifiedTopology: true,
    });
    await client.connect();
    console.log("Connected to MongoDB");
  }
  return client;
}

const buildAutocompletePipeline = (query, indexName, path) => {
  const pipeline = [];

  pipeline.push({
    $search: {
      index: indexName,
      autocomplete: {
        query: query,
        path: path,
        fuzzy: {
          maxEdits: 2,
          prefixLength: 2,
        },
      },
    },
  });

  pipeline.push({
    $match: {
      $or: [
        { stockStatus: { $exists: false } },
        { stockStatus: "instock" }
      ],
    },
  });

  pipeline.push(
    { $limit: 10 },
    {
      $project: {
        _id: 0,
        suggestion: `$${path}`,
        score: { $meta: "searchScore" },
        url: 1,
        image: 1,
        price: 1,
      },
    }
  );

  return pipeline;
};

app.get("/autocomplete", async (req, res) => {
  const { dbName, collectionName1, collectionName2, query } = req.query;

  if (!dbName || !collectionName1 || !collectionName2 || !query) {
    return res.status(400).json({ error: "Database name, both collection names, and query are required." });
  }

  try {
    const client = await connectToMongoDB(mongodbUri);
    const db = client.db(dbName);
    const collection1 = db.collection(collectionName1);
    const collection2 = db.collection(collectionName2);

    const pipeline1 = buildAutocompletePipeline(query, "default", "name");
    const pipeline2 = buildAutocompletePipeline(query, "default2", "query");

    const [suggestions1, suggestions2] = await Promise.all([
      collection1.aggregate(pipeline1).toArray(),
      collection2.aggregate(pipeline2).toArray(),
    ]);

    const labeledSuggestions1 = suggestions1.map(item => ({ 
      suggestion: item.suggestion, 
      score: item.score,
      source: collectionName1,
      url: item.url,
      price: item.price,
      image: item.image
    }));

    const labeledSuggestions2 = suggestions2.map(item => ({ 
      suggestion: item.suggestion, 
      score: item.score,
      source: collectionName2,
      url: item.url
    }));

    const combinedSuggestions = [...labeledSuggestions1, ...labeledSuggestions2]
      .sort((a, b) => b.score - a.score)
      .filter((item, index, self) =>
        index === self.findIndex((t) => t.suggestion === item.suggestion)
      );

    res.json(combinedSuggestions);
  } catch (error) {
    console.error("Error fetching autocomplete suggestions:", error);
    res.status(500).json({ error: "Server error" });
  }
});

const buildFuzzySearchPipeline = (cleanedHebrewText, filters) => {
  const pipeline = [
    {
      $search: {
        index: "default",
        compound: {
          should: [
            {
              text: {
                query: cleanedHebrewText,
                path: "name",
                fuzzy: {
                  maxEdits: 2,
                  prefixLength: 1,
                  maxExpansions: 50,
                },
              },
            },
            {
              text: {
                query: cleanedHebrewText,
                path: "description",
                fuzzy: {
                  maxEdits: 2,
                  prefixLength: 1,
                  maxExpansions: 50,
                },
              },
            },
          ],
        },
      },
    },
  ];

  if (filters && Object.keys(filters).length > 0) {
    const matchStage = {};

    if (filters.category != null) {
      matchStage.category = Array.isArray(filters.category)
        ? { $in: filters.category }
        : filters.category;
    }

    if (filters.type != null) {
      matchStage.type = { $regex: filters.type, $options: "i" };
    }

    if (filters.minPrice != null && filters.maxPrice != null) {
      matchStage.price = { $gte: filters.minPrice, $lte: filters.maxPrice };
    } else if (filters.minPrice != null) {
      matchStage.price = { $gte: filters.minPrice };
    } else if (filters.maxPrice != null) {
      matchStage.price = { $lte: filters.maxPrice };
    }

    if (filters.price != null) {
      const price = filters.price;
      const priceRange = price * 0.15;
      matchStage.price = { $gte: price - priceRange, $lte: price + priceRange };
    }

    matchStage.$or = [
      { stockStatus: { $exists: false } },
      { stockStatus: "instock" },
    ];

    if (Object.keys(matchStage).length > 0) {
      pipeline.push({ $match: matchStage });
    }
  }

  pipeline.push({ $limit: 10 });

  return pipeline;
};

function buildVectorSearchPipeline(queryEmbedding, filters = {}) {
  const filter = {};

  if (filters.category) {
    filter.category = Array.isArray(filters.category)
      ? { $in: filters.category }
      : filters.category;
  }

  if (filters.type) {
    filter.type = filters.type;
  }

  if (filters.minPrice && filters.maxPrice) {
    filter.price = { $gte: filters.minPrice, $lte: filters.maxPrice };
  } else if (filters.minPrice) {
    filter.price = { $gte: filters.minPrice };
  } else if (filters.maxPrice) {
    filter.price = { $lte: filters.maxPrice };
  }

  if (filters.price) {
    const price = filters.price;
    const priceRange = price * 0.15;
    filter.price = { $gte: price - priceRange, $lte: price + priceRange };
  }

  const pipeline = [
    {
      $vectorSearch: {
        index: "vector_index",
        path: "embedding",
        queryVector: queryEmbedding,
        exact: true,
        limit: 16,
        ...(Object.keys(filter).length && { filter }),
      },
    },
  ];
  
  const postMatchClauses = [];
  postMatchClauses.push({
    $or: [
      { stockStatus: "instock" },
      { stockStatus: { $exists: false } },
    ],
  });

  if (postMatchClauses.length > 0) {
    pipeline.push({ $match: { $and: postMatchClauses } });
  }

  return pipeline;
}

async function isHebrew(query) {
  const hebrewPattern = /[\u0590-\u05FF]/;
  return hebrewPattern.test(query);
}

async function translateQuery(query, context) {
  try {
    const needsTranslation = await isHebrew(query);

    if (!needsTranslation) {
      return query;
    }

    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content:
            `Translate the following text from Hebrew to English. If it's already in English, keep it in English and don't translate it to Hebrew. The context is a search query in ${context}, so you probably get words attached to products or their descriptions. Respond with the answer only, without explanations. Pay attention to the word שכלי or שאבלי- those are meant to be chablis.`
        },
        { role: "user", content: query },
      ],
    });

    const translatedText = response.choices[0]?.message?.content?.trim();
    console.log("Translated query:", translatedText);
    return translatedText;
  } catch (error) {
    console.error("Error translating query:", error);
    throw error;
  }
}

function removeWineFromQuery(translatedQuery, noWord) {
  if (!noWord) return translatedQuery;

  const queryWords = translatedQuery.split(" ");
  const filteredWords = queryWords.filter((word) => {
    return !noWord.includes(word.toLowerCase());
  });

  return filteredWords.join(" ");
}

function removeWordsFromQuery(query, noHebrewWord) {
  if (!noHebrewWord) return query;

  const queryWords = query.split(" ");
  const filteredWords = queryWords.filter((word) => {
    return !noHebrewWord.includes(word.toLowerCase() && isNaN(word));
  });

  return filteredWords.join(" ");
}

async function extractFiltersFromQuery(query, categories, types, example) {
  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4o",
      response_format: { type: "json_object" },
      messages: [
        { role: "system", content: `Extract the following filters from the query if they exist:
                  1. price (exact price, indicated by the words 'ב' or 'באיזור ה-').
                  2. minPrice (minimum price, indicated by 'החל מ' or 'מ').
                  3. maxPrice (maximum price, indicated by the word 'עד').
                  4. category - one of the following Hebrew words: ${categories}. Pay close attention to find these categories in the query, and look if the user mentions a shortened version (e.g., 'רוזה' instead of 'יין רוזה').
                  5. type - one or both of the following Hebrew words: ${types}. Pay close attention to find these types in the query.
                Return the extracted filters in JSON format. If a filter is not present in the query, omit it from the JSON response. For example:
               ${example}.` },
        { role: "user", content: query },
      ],
      temperature: 0.5,
    });

    const content = response.choices[0]?.message?.content;
    const filters = JSON.parse(content);
    console.log("Extracted filters:", filters);
    return filters;
  } catch (error) {
    console.error("Error extracting filters:", error);
    throw error;
  }
}

async function getQueryEmbedding(cleanedText) {
  try {
    const response = await openai.embeddings.create({
      model: "text-embedding-3-large",
      input: cleanedText,
    });
    return response.data[0]?.embedding || null;
  } catch (error) {
    console.error("Error fetching query embedding:", error);
    throw error;
  }
}

async function logQuery(queryCollection, query, filters) {
  const timestamp = new Date();

  const entity = `${filters.category || "unknown"} ${filters.type || "unknown"}`;

  const queryDocument = {
    query: query,
    timestamp: timestamp,
    category: filters.category || "unknown",
    price: filters.price || "unknown",
    minPrice: filters.minPrice || "unknown",
    maxPrice: filters.maxPrice || "unknown",
    type: filters.type || "unknown",
    entity: entity.trim(),
  };

  await queryCollection.insertOne(queryDocument);
}

async function reorderResultsWithGPT(
  combinedResults,
  translatedQuery,
  query,
  alreadyDelivered = []
) {
  try {
    if (!Array.isArray(alreadyDelivered)) {
      alreadyDelivered = [];
    }
    const filteredResults = combinedResults.filter(
      (product) => !alreadyDelivered.includes(product._id.toString())
    );

    const productData = filteredResults.map((product) => ({
      id: product._id.toString(),
      name: product.name || "No name",
      description: product.description1 || "No description",
    }));

    console.log(JSON.stringify(productData.slice(0, 4)));

    const messages = [
      {
        role: "user",
        parts: [{ text: `You are an advanced AI model specializing in e-commerce queries. Your role is to analyze a given "${translatedQuery}" from an e-commerce site, along with a provided list of products (each including a name and description), and return the **most relevant product IDs** based solely on how well the product names and descriptions match the query.

### Key Instructions:

1. Ignore pricing details (already filtered).
2. Output must be a plain array of IDs, no extra text.
3. ONLY return the most relevant products related to the query ranked in the right order, but never more that 10.

` }],
      },
      {
        role: "user",
        parts: [{ text: JSON.stringify(productData, null, 4) }],
      },
    ];
    
     const geminiResponse = await model.generateContent({
          contents: messages,
        });


    const response = await geminiResponse.response;
   const  reorderedText = response.text();
    console.log("Reordered IDs text:", reorderedText);

    if (!reorderedText) {
      throw new Error("No content returned from Gemini");
    }

      const cleanedText = reorderedText
      .trim()
      .replace(/[^,\[\]"'\w]/g, "")
      .replace(/json/gi, "");
    try {
        const reorderedIds = JSON.parse(cleanedText);
        if (!Array.isArray(reorderedIds)) {
            throw new Error("Invalid response format from Gemini. Expected an array of IDs.");
        }
        return reorderedIds;
    } catch (parseError) {
      console.error(
        "Failed to parse Gemini response:",
        parseError,
        "Cleaned Text:",
        cleanedText
      );
      throw new Error("Response from Gemini could not be parsed as a valid array.");
    }
  } catch (error) {
    console.error("Error reordering results with Gemini:", error);
    throw error;
  }
}



async function reorderImagesWithGPT(
 combinedResults,
 translatedQuery,
 query,
 alreadyDelivered = []
) {
 try {
   if (!Array.isArray(alreadyDelivered)) {
     alreadyDelivered = [];
   }

   const filteredResults = combinedResults.filter(
     (product) => !alreadyDelivered.includes(product._id.toString())
   );

   const productData = combinedResults.map(product => ({
     id: product._id.toString(),
     name: product.name,
     image: product.image,
     description: product.description1,
   }));

   const imagesToSend = combinedResults.map(product => ({
     imageUrl: product.image || "No image"
   }));


   const messages = [
     {
       role: "user",
       parts: [
         {
           text: `You are an advanced AI model specializing in e-commerce queries. Your role is to analyze a given "${translatedQuery}", from an e-commerce site, along with a provided list of products (each including only an image), and return the **most relevant product IDs** based on how well the product images match the query.

### Key Instructions:
1. Ignore pricing details (already filtered).
2. Output must be a JSON array of IDs, with no extra text or formatting.
3. Rank strictly according to the product images.
4. Return at least 5 but no more than 8 product IDs.
5. Answer ONLY with the Array, do not add any other text beside it- NEVER!

example: [ "id1", "id2", "id3", "id4" ]

`,
         },
         {
             text:  JSON.stringify(productData, null, 2),
           },
         {
           text: JSON.stringify(
             {
               type: "image_url",
               images: imagesToSend,
             },
             null,
             4
           ),
         },
         ]
     },
   ];

 



   const geminiResponse = await model.generateContent({
     contents: messages,
   });


     const responseText = geminiResponse.response.text()
     console.log("Gemini Reordered IDs text:", responseText);
 


   if (!responseText) {
     throw new Error("No content returned from Gemini");
   }

  // If you want usage details:
   // console.log(geminiResponse.usage);


   const cleanedText = responseText
   .trim()
   .replace(/[^,\[\]"'\w]/g, "")
   .replace(/json/gi, "");


   try {
     const reorderedIds = JSON.parse(cleanedText);
     if (!Array.isArray(reorderedIds)) {
       throw new Error("Invalid response format from Gemini. Expected an array of IDs.");
     }
     return reorderedIds;
   } catch (parseError) {
     console.error(
       "Failed to parse Gemini response:",
       parseError,
       "Cleaned Text:",
       cleanedText
     );
     throw new Error("Response from Gemini could not be parsed as a valid array.");
   }
 } catch (error) {
   console.error("Error reordering results with Gemini:", error);
   throw error;
 }
}

async function getProductsByIds(ids, dbName, collectionName) {
  if (!ids || !Array.isArray(ids)) {
    console.error("getProductsByIds: ids is not an array", ids);
    return []; // or throw an error if that's preferred
  }
  try {
    const client = await connectToMongoDB(mongodbUri);
    const db = client.db(dbName);
    const collection = db.collection(collectionName);

    const objectIdArray = ids
      .map((id) => {
        try {
          return new ObjectId(id);
        } catch (error) {
          console.error(`Invalid ObjectId format: ${id}`);
          return null;
        }
      })
      .filter((id) => id !== null);

    const products = await collection
      .find({ _id: { $in: objectIdArray } })
      .toArray();

    const orderedProducts = ids
      .map((id) => products.find((p) => p && p._id.toString() === id))
      .filter((product) => product !== undefined);

    console.log(`Number of products returned: ${orderedProducts.length}/${ids.length}`);
    return orderedProducts;
  } catch (error) {
    console.error("Error fetching products by IDs:", error);
    throw error;
  }
}


app.post("/search", async (req, res) => {
  const {
    dbName,
    collectionName,
    query,
    categories,
    types,
    example,
    noWord,
    noHebrewWord,
    context,
    useImages,
  } = req.body;

  if (!query || !dbName || !collectionName) {
    return res.status(400).json({
      error: "Query, database name, and collection name are required",
    });
  }

  try {
    const client = await connectToMongoDB(mongodbUri);
    const db = client.db(dbName);
    const collection = db.collection(collectionName);
    const querycollection = db.collection("queries");

    const translatedQuery = await translateQuery(query, context);
    if (!translatedQuery)
      return res.status(500).json({ error: "Error translating query" });

    const cleanedText = removeWineFromQuery(translatedQuery, noWord);
    console.log("Cleaned query for embedding:", cleanedText);

    const filters = await extractFiltersFromQuery(query, categories, types, example);

    logQuery(querycollection, query, filters);

    const queryEmbedding = await getQueryEmbedding(cleanedText);
    if (!queryEmbedding)
      return res.status(500).json({ error: "Error generating query embedding" });

    const FUZZY_WEIGHT = 1;
    const VECTOR_WEIGHT = 1;
    const RRF_CONSTANT = 60;

    function calculateRRFScore(fuzzyRank, vectorRank, VECTOR_WEIGHT) {
      return (
        FUZZY_WEIGHT * (1 / (RRF_CONSTANT + fuzzyRank)) +
        VECTOR_WEIGHT * (1 / (RRF_CONSTANT + vectorRank))
      );
    }

    const cleanedHebrewText = removeWordsFromQuery(query, noHebrewWord);
    console.log("Cleaned query for fuzzy search:", cleanedHebrewText);

    const fuzzySearchPipeline = buildFuzzySearchPipeline(cleanedHebrewText, filters);
    const fuzzyResults = await collection.aggregate(fuzzySearchPipeline).toArray();

    const vectorSearchPipeline = buildVectorSearchPipeline(queryEmbedding, filters);
    const vectorResults = await collection.aggregate(vectorSearchPipeline).toArray();

    const documentRanks = new Map();

    fuzzyResults.forEach((doc, index) => {
      documentRanks.set(doc._id.toString(), {
        fuzzyRank: index,
        vectorRank: Infinity,
      });
    });

    vectorResults.forEach((doc, index) => {
      const existingRanks = documentRanks.get(doc._id.toString()) || {
        fuzzyRank: Infinity,
        vectorRank: Infinity,
      };
      documentRanks.set(doc._id.toString(), {
        ...existingRanks,
        vectorRank: index,
      });
    });

    const combinedResults = Array.from(documentRanks.entries())
      .map(([id, ranks]) => {
        const doc =
          fuzzyResults.find((d) => d._id.toString() === id) ||
          vectorResults.find((d) => d._id.toString() === id);
        return {
          ...doc,
          rrf_score: calculateRRFScore(ranks.fuzzyRank, ranks.vectorRank, VECTOR_WEIGHT),
        };
      })
      .sort((a, b) => b.rrf_score - a.rrf_score);

      let reorderedIds;
      try {
        const reorderFn = useImages ? reorderImagesWithGPT : reorderResultsWithGPT;
        reorderedIds = await reorderFn(combinedResults, translatedQuery, query);
      } catch (error) {
        console.error("LLM reordering failed, falling back to default ordering:", error);
        // Fallback: use the combinedResults order (which is already ranked by RRF)
        reorderedIds = combinedResults.map((result) => result._id.toString());
      }
      // 8) Retrieve final product docs in the GPT order
      const orderedProducts = await getProductsByIds(reorderedIds, dbName, collectionName);
  
      // 9) Combine GPT-ordered results + leftover
      const reorderedProductIds = new Set(reorderedIds);
      const remainingResults = combinedResults.filter(
        (r) => !reorderedProductIds.has(r._id.toString())
      );
  
      const formattedResults = [
        ...orderedProducts.map((product) => ({
          id: product._id.toString(),
          name: product.name,
          description: product.description,
          price: product.price,
          image: product.image,
          url: product.url,
          highlight: true,
          onSale: product.onSale
        })),
        ...remainingResults.map((r) => ({
          id: r._id.toString(),
          name: r.name,
          description: r.description,
          price: r.price,
          image: r.image,
          url: r.url,
          onSale:r.onSale
        })),
      ];
  
      res.json(formattedResults);
    } catch (error) {
      console.error("Error handling search request:", error);
      if (!res.headersSent) {
        res.status(500).json({ error: "Server error." });
      }
    } 
  });

app.get("/products", async (req, res) => {
  const { dbName, collectionName, limit = 10 } = req.query;

  if (!dbName || !collectionName) {
    return res.status(400).json({
      error: "Database name and collection name are required",
    });
  }

  try {
    const client = await connectToMongoDB(mongodbUri);
    const db = client.db(dbName);
    const collection = db.collection(collectionName);

    const products = await collection.find().limit(Number(limit)).toArray();

    const results = products.map((product) => ({
      id: product._id,
      name: product.name,
      description: product.description,
      price: product.price,
      image: product.image,
      url: product.url,
    }));

    res.json(results);
  } catch (error) {
    console.error("Error fetching products:", error);
    res.status(500).json({ error: "Server error" });
  }
});

app.post("/recommend", async (req, res) => {
  const { productName, dbName, collectionName } = req.body;

  if (!productName) {
    return res.status(400).json({ error: "Product name is required" });
  }

  try {
    const client = await connectToMongoDB(mongodbUri);
    const db = client.db(dbName);
    const collection = db.collection(collectionName);

    const product = await collection.findOne({ name: productName });

    if (!product) {
      return res.status(404).json({ error: "Product not found" });
    }

    const { embedding, price } = product;
    const minPrice = price * 0.9;
    const maxPrice = price * 1.1;

    const pipeline = [
      {
        $vectorSearch: {
          index: "vector_index",
          path: "embedding",
          queryVector: embedding,
          numCandidates: 100,
          limit: 10,
        },
      },
      {
        $match: {
          price: { $gte: minPrice, $lte: maxPrice },
        },
      },
    ];

    const similarProducts = await collection.aggregate(pipeline).toArray();

    const results = similarProducts.map((product) => ({
      id: product._id,
      name: product.name,
      description: product.description,
      price: product.price,
      image: product.image,
      url: product.url,
      rrf_score: product.rrf_score,
    }));

    res.json(results);
  } catch (error) {
    console.error("Error fetching recommendations:", error);
    res.status(500).json({ error: "Server error" });
  }
});

const PORT = process.env.PORT || 8000;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});

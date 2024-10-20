import express from "express";
import bodyParser from "body-parser";
import { MongoClient, ObjectId } from "mongodb";
import { OpenAI } from "openai";
import cors from "cors";
import dotenv from "dotenv";

dotenv.config();

const app = express();
app.use(bodyParser.json());
app.use(cors({ origin: "*" }));

// Initialize OpenAI client
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const mongodbUri = process.env.MONGODB_URI;
let client;


async function connectToMongoDB(mongodbUri) {
  if (!client || !client.topology || !client.topology.isConnected()) { 
    client = new MongoClient(mongodbUri, {
      useNewUrlParser: true,
      useUnifiedTopology: true,
    });
    await client.connect();
  }
  return client;
}

const buildFuzzySearchPipeline = (cleanedHebrewText, filters) => {
  const pipeline = [
    {
      $search: {
        index: "default",
        text: {
          query: cleanedHebrewText,
          path: ["name","description"],
          fuzzy: {
            maxEdits: 2,
            prefixLength: 0,
            maxExpansions: 50,
          },
        },
      },
    },
  ];

  if (filters && Object.keys(filters).length > 0) {
    const matchStage = {};

    if (filters.category ?? null) {
      // Check if the category is an array or a single string
      matchStage.category = Array.isArray(filters.category)
        ? { $in: filters.category } // If it's an array, use $in
        : filters.category; // Otherwise, just match the single value
    }
    if (filters.type ?? null) {
      matchStage.type = { $regex: filters.type, $options: "i" };
    }
    if (filters.minPrice && filters.maxPrice) {
      matchStage.price = { $gte: filters.minPrice, $lte: filters.maxPrice };
    } else if (filters.minPrice) {
      matchStage.price = { $gte: filters.minPrice };
    } else if (filters.maxPrice) {
      matchStage.price = { $lte: filters.maxPrice };
    }
    if (filters.price) {
        const price = filters.price;
        const priceRange = price * 0.15; // 15% of the price
        matchStage.price = { $gte: price - priceRange, $lte: price + priceRange };
      }

    if (Object.keys(matchStage).length > 0) {
      pipeline.push({ $match: matchStage });
    }
  }

  pipeline.push({ $limit: 20 }); // Increase limit for better RRF results

  return pipeline;
};

const buildVectorSearchPipeline = (queryEmbedding, filters) => {
  const pipeline = [
    {
      $vectorSearch: {
        index: "vector_index",
        path: "embedding",
        queryVector: queryEmbedding,
        numCandidates: 300,
        limit: 50, // Increase limit for better RRF results
      },
    },
  ];

  if (filters && Object.keys(filters).length > 0) {
    const matchStage = {};

    if (filters.category ?? null) {
      // Check if the category is an array or a single string
      matchStage.category = Array.isArray(filters.category)
        ? { $in: filters.category } // If it's an array, use $in
        : filters.category; // Otherwise, just match the single value
    }
    if (filters.type ?? null) {
      matchStage.type = { $regex: filters.type, $options: "i" };
    }
    if (filters.minPrice && filters.maxPrice) {
      matchStage.price = { $gte: filters.minPrice, $lte: filters.maxPrice };
    } else if (filters.minPrice) {
      matchStage.price = { $gte: filters.minPrice };
    } else if (filters.maxPrice) {
      matchStage.price = { $lte: filters.maxPrice };
    }
    if (filters.price) {
        const price = filters.price;
        const priceRange = price * 0.15; // 15% of the price
        matchStage.price = { $gte: price - priceRange, $lte: price + priceRange };
      }

    if (Object.keys(matchStage).length > 0) {
      pipeline.push({ $match: matchStage });
    }
  }

  return pipeline;
};

// Utility function to translate query from Hebrew to English
async function isHebrew(query) {
  // Hebrew characters range in Unicode
  const hebrewPattern = /[\u0590-\u05FF]/;
  return hebrewPattern.test(query);
}

async function translateQuery(query, context) {
  try {
    // Check if the query is in Hebrew
    const needsTranslation = await isHebrew(query);

    if (!needsTranslation) {
      // If the query is already in English, return it as is
      return query;
    }

    // Proceed with translation if the query is in Hebrew
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

// New function to remove words from the query
function removeWineFromQuery(translatedQuery, noWord) {
  if (!noWord) return translatedQuery;

  const queryWords = translatedQuery.split(" ");
  const filteredWords = queryWords.filter((word) => {
    // Remove the word if it's in the noWord list or if it's a number
    return !noWord.includes(word.toLowerCase()) && isNaN(Number(word));
  });

  return filteredWords.join(" ");
}

function removeWordsFromQuery(query, noHebrewWord) {
  if (!noHebrewWord) return query;

  const queryWords = query.split(" ");
  const filteredWords = queryWords.filter((word) => {
    // Remove the word if it's in the noWords list or if it's a number
    return !noHebrewWord.includes(word.toLowerCase()) && isNaN(Number(word));
  });

  return filteredWords.join(" ");
}

// Utility function to extract filters from query using LLM
async function extractFiltersFromQuery(query, systemPrompt) {
  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4o",
      response_format: { type: "json_object" },
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: query },
      ],
      temperature: 0.5,
    });

    const content = response.choices[0]?.message?.content;
    const filters = JSON.parse(content);
    console.log("Extracted filters:", filters);
    console.log(Object.keys(filters).length);

    return filters;
  } catch (error) {
    console.error("Error extracting filters:", error);
    throw error;
  }
}

// Utility function to get the embedding for a query
async function getQueryEmbedding(cleanedText) {
  try {
    // Remove 'wine' from the translated text

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
  const timestamp = new Date(); // Current timestamp

  // Combine filters.category and filters.type to form the 'entity'
  const entity = `${filters.category || "unknown"} ${
    filters.type || "unknown"
  }`;

  // Build the query document to insert
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

  // Insert the query document into the queries collection
  await queryCollection.insertOne(queryDocument);
}


async function reorderResultsWithGPT(combinedResults, query) {
  try {
    // Prepare an array of objects containing only product IDs and descriptions
    const productData = combinedResults.map((product) => ({
      id: product._id.toString(),
      description: product.description || "No description",
    }));

    const messages = [
      {
        role: "user",
        content: `Here is a search query: "${query}". Please reorder the following products based on their descriptions' and names relevance to the query. Return the reordered list as an array of product IDs in the order they should appear. answer only with the array of product IDs (no 'json' at the beginning or something, just plain array- always!) in the right order, nothing else.`,
      },
      {
        role: "user",
        content: JSON.stringify(productData, null, 2), // Send only ID and description
      },
    ];

    // Send the request to GPT-4
    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini", // Use GPT-4, or "gpt-3.5-turbo" for faster response
      messages: messages,
      temperature: 0.7,
    });

    // Extract and parse the reordered product IDs
    const reorderedText = response.choices[0]?.message?.content;
   
    
    if (!reorderedText) {
      throw new Error("No content returned from GPT-4");
    }

    // Parse the response which should be an array of product IDs
    const reorderedIds = JSON.parse(reorderedText);
    
    if (!Array.isArray(reorderedIds)) {
      throw new Error("Invalid response format from GPT-4. Expected an array of IDs.");
    }

    return reorderedIds;
  } catch (error) {
    console.error("Error reordering results with GPT:", error);
    throw error;
  }
}



async function getProductsByIds(ids, dbName, collectionName) {
  try {
    // Ensure the MongoDB client is connected
    const client = await connectToMongoDB(mongodbUri);
    const db = client.db(dbName);
    const collection = db.collection(collectionName);

    // Convert the ids to ObjectId and filter out invalid ones
    const objectIdArray = ids.map(id => {
      try {
        return new ObjectId(id); // Convert to ObjectId
      } catch (error) {
        console.error(`Invalid ObjectId format: ${id}`);
        return null; // If not valid, return null
      }
    }).filter(id => id !== null); // Filter out invalid ObjectIds

    // Fetch products from MongoDB
    const products = await collection.find({ _id: { $in: objectIdArray } }).toArray();

    // Log fetched products for debugging
   
    // Map the products back to the original IDs order, skipping undefined products
    const orderedProducts = ids.map(id => products.find(product => product && product._id.toString() === id))
                               .filter(product => product !== undefined); // Skip missing products

    // Log if any products were skipped
    console.log(`Number of products returned: ${orderedProducts.length}/${ids.length}`);

    return orderedProducts;
  } catch (error) {
    console.error("Error fetching products by IDs:", error);
    throw error;
  }
}




// Route to handle the search endpoint
app.post("/search", async (req, res) => {
  const {
    dbName,
    collectionName,
    query,
    systemPrompt,
    noWord,
    noHebrewWord,
    context
  } = req.body;

  if (!query || !mongodbUri || !dbName || !collectionName || !systemPrompt) {
    return res.status(400).json({
      error:
        "Query, MongoDB URI, database name, collection name, and system prompt are required",
    });
  }

  let client;

  try {
    client = await connectToMongoDB(mongodbUri);
    const db = client.db(dbName);
    const collection = db.collection(collectionName);
    const querycollection = db.collection("queries");

    // Translate query
    const translatedQuery = await translateQuery(query, context);
    if (!translatedQuery)
      return res.status(500).json({ error: "Error translating query" });

    const cleanedText = removeWineFromQuery(translatedQuery, noWord);
    console.log(noWord);
    console.log("Cleaned query for embedding:", cleanedText);
    // Extract filters from the translated query
    const filters = await extractFiltersFromQuery(query, systemPrompt);

    logQuery(querycollection, query, filters);

    // Get query embedding
    const queryEmbedding = await getQueryEmbedding(cleanedText);
    if (!queryEmbedding)
      return res
        .status(500)
        .json({ error: "Error generating query embedding" });

    const RRF_CONSTANT = 60;
    const VECTOR_WEIGHT = 1;

    function calculateRRFScore(fuzzyRank, vectorRank, VECTOR_WEIGHT) {
      return (
        1 / (RRF_CONSTANT + fuzzyRank) +
        VECTOR_WEIGHT * (1 / (RRF_CONSTANT + vectorRank))
      );
    }

    // Perform fuzzy search
    const cleanedHebrewText = removeWordsFromQuery(query, noHebrewWord);
    console.log(noHebrewWord);
    console.log("Cleaned query for fuzzy search:", cleanedHebrewText); // Check if cleanedText
    const fuzzySearchPipeline = buildFuzzySearchPipeline(
      cleanedHebrewText,
      filters
    );
    const fuzzyResults = await collection
      .aggregate(fuzzySearchPipeline)
      .toArray();

    // Perform vector search
    const vectorSearchPipeline = buildVectorSearchPipeline(
      queryEmbedding,
      filters
    );
    const vectorResults = await collection
      .aggregate(vectorSearchPipeline)
      .toArray();

    // Create a map to store the best rank for each document
    const documentRanks = new Map();

    // Process fuzzy search results
    fuzzyResults.forEach((doc, index) => {
      documentRanks.set(doc._id.toString(), {
        fuzzyRank: index,
        vectorRank: Infinity,
      });
    });

    // Process vector search results
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

    // Calculate RRF scores and create the final result set
    const combinedResults = Array.from(documentRanks.entries())
      .map(([id, ranks]) => {
        const doc =
          fuzzyResults.find((d) => d._id.toString() === id) ||
          vectorResults.find((d) => d._id.toString() === id);
        return {
          ...doc,
          rrf_score: calculateRRFScore(
            ranks.fuzzyRank,
            ranks.vectorRank,
            VECTOR_WEIGHT
          ),
        };
      })
      .sort((a, b) => b.rrf_score - a.rrf_score)
      .slice(0, 12); // Get the top 12 results

    // Reorder the results with GPT-4 based on description relevance to the query
    const reorderedIds = await reorderResultsWithGPT(combinedResults, query);
    const orderedProducts = await getProductsByIds(reorderedIds, dbName, collectionName);

    // Format results
  

    const formattedResults = orderedProducts.map((product) => ({
      id: product._id.toString(),
      name: product.name,
      description: product.description,
      price: product.price,
      image: product.image,
      url: product.url,
    }));

    // Ensure that the response is sent only once
    res.json(formattedResults);
  } catch (error) {
    console.error("Error handling search request:", error);
    // If an error occurs, send the response here and ensure you don't send it again later
    if (!res.headersSent) {
      res.status(500).json({ error: "Server error." });
    }
  }
});



app.get("/products", async (req, res) => {
  const { dbName, collectionName, limit = 10 } = req.query;

  if (!dbName || !collectionName) {
    return res.status(400).json({
      error: "MongoDB URI, database name, and collection name are required",
    });
  }

  let client;

  try {
    const client = await connectToMongoDB(mongodbUri);
    const db = client.db(dbName);
    const collection = db.collection(collectionName);

    // Fetch a default set of products, e.g., the latest 10 products
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
  } finally {
    if (client) {
      await client.close();
    }
  }
});
app.post("/recommend", async (req, res) => {
  const { productName, dbName, collectionName } = req.body;

  if (!productName) {
    return res.status(400).json({ error: "Product URL is required" });
  }

  let client;

  try {
    client = await connectToMongoDB(mongodbUri);
    const db = client.db(dbName);
    const collection = db.collection(collectionName);

    // Find the product by URL
    const product = await collection.findOne({ name: productName });

    if (!product) {
      return res.status(404).json({ error: "Product not found" });
    }

    // Extract the embedding and price range from the product
    const { embedding, price } = product;

    // Define a price range (e.g., ±10% of the product's price)
    const minPrice = price * 0.9;
    const maxPrice = price * 1.1;

    // Build the pipeline to find similar products based on embedding and price range
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
  } finally {
    if (client) {
      await client.close();
    }
  }
});

const PORT = process.env.PORT || 8000;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
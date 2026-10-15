import { HfInference } from "@huggingface/inference";
import type { Pinecone } from "@pinecone-database/pinecone";
import { modelname } from "@/app/config";

type QueryOptions = {
  reportId?: string;
  topK?: number;
  minScore?: number;
};

const embeddingModel = process.env.EMBEDDING_MODEL || modelname;
const embeddingCache = new Map<string, number[]>();
const maxCacheEntries = 1000;

function cacheEmbedding(text: string, vector: number[]) {
  embeddingCache.delete(text);
  embeddingCache.set(text, vector);
  if (embeddingCache.size > maxCacheEntries) {
    const oldestKey = embeddingCache.keys().next().value;
    if (oldestKey !== undefined) embeddingCache.delete(oldestKey);
  }
}

function normalizeEmbedding(value: unknown): number[] {
  if (!Array.isArray(value) || value.length === 0) {
    throw new Error("Embedding provider returned an empty vector.");
  }

  if (value.every((item): item is number => typeof item === "number")) {
    return value;
  }

  const rows = value
    .map((item) => Array.isArray(item) ? item : [])
    .filter((row): row is number[] => row.length > 0 && row.every((item) => typeof item === "number"));

  if (rows.length === 0) {
    throw new Error("Embedding provider returned an invalid vector.");
  }

  if (rows.length === 1) return rows[0];

  const vector = rows[0].map((_, column) =>
    rows.reduce((sum, row) => sum + (row[column] ?? 0), 0) / rows.length
  );
  const magnitude = Math.sqrt(vector.reduce((sum, item) => sum + item * item, 0));
  if (magnitude === 0) throw new Error("Embedding provider returned a zero vector.");
  return vector.map((item) => item / magnitude);
}

async function getEmbeddings(texts: string[]): Promise<number[][]> {
  const vectors: Array<number[] | undefined> = texts.map((text) => embeddingCache.get(text));
  const missingIndexes = vectors.flatMap((vector, index) => vector ? [] : [index]);

  if (missingIndexes.length > 0) {
    const missingTexts = missingIndexes.map((index) => texts[index]);
    let generated: number[][] | undefined;
    const embeddingUrl = process.env.MCP_EMBED_URL
      || `http://${process.env.MCP_EMBED_HOST || "127.0.0.1"}:${process.env.MCP_EMBED_PORT || "8787"}/embed`;

    try {
      const response = await fetch(embeddingUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ texts: missingTexts }),
        signal: AbortSignal.timeout(15_000),
      });
      if (!response.ok) throw new Error(`Embedding server returned HTTP ${response.status}.`);

      const result: { success?: boolean; embeddings?: unknown[]; error?: string } = await response.json();
      if (!result.success || !Array.isArray(result.embeddings) || result.embeddings.length !== missingTexts.length) {
        throw new Error(result.error || "Embedding server returned an invalid response.");
      }
      generated = result.embeddings.map(normalizeEmbedding);
    } catch (error) {
      console.warn("Local embedding server unavailable; trying Hugging Face.", error);
    }

    if (!generated) {
      if (!process.env.HF_TOKEN) {
        throw new Error("No embedding provider is available. Start the local embedding server or configure HF_TOKEN.");
      }

      const hf = new HfInference(process.env.HF_TOKEN);
      const result = await hf.featureExtraction({
        model: embeddingModel,
        inputs: missingTexts,
      });
      if (!Array.isArray(result) || result.length !== missingTexts.length) {
        throw new Error("Hugging Face returned an invalid embedding response.");
      }
      generated = result.map(normalizeEmbedding);
    }

    for (let index = 0; index < missingIndexes.length; index++) {
      const originalIndex = missingIndexes[index];
      const vector = generated[index];
      vectors[originalIndex] = vector;
      cacheEmbedding(texts[originalIndex], vector);
    }
  }

  return vectors.map((vector) => {
    if (!vector) throw new Error("Failed to generate an embedding.");
    return vector;
  });
}

function splitIntoChunks(text: string, maxLength = 1200): string[] {
  const normalized = text.trim();
  if (!normalized) return [];

  const chunks: string[] = [];
  const overlap = 150;
  for (let start = 0; start < normalized.length; start += maxLength - overlap) {
    const chunk = normalized.slice(start, start + maxLength).trim();
    if (chunk) chunks.push(chunk);
  }
  return chunks;
}

export async function createAndStoreVectorEmbeddings(
  pinecone: Pinecone,
  indexName: string,
  namespace: string,
  text: string,
  reportId: string
): Promise<boolean> {
  try {
    const chunks = splitIntoChunks(text);
    if (chunks.length === 0) throw new Error("There is no report text to embed.");

    const vectors = await getEmbeddings(chunks);
    const index = pinecone.index(indexName).namespace(namespace);
    const batchSize = 100;

    for (let start = 0; start < chunks.length; start += batchSize) {
      const records = chunks.slice(start, start + batchSize).map((chunk, offset) => {
        const chunkIndex = start + offset;
        return {
          id: `${reportId}_${chunkIndex}`,
          values: vectors[chunkIndex],
          metadata: { text: chunk, reportId, chunkIndex },
        };
      });
      await index.upsert(records);
    }

    return true;
  } catch (error) {
    console.error("Failed to create and store report embeddings:", error);
    return false;
  }
}

export async function queryPineconeVectorStore(
  pinecone: Pinecone,
  indexName: string,
  namespace: string,
  query: string,
  options: QueryOptions = {}
): Promise<string> {
  const [vector] = await getEmbeddings([query]);
  const filter = options.reportId ? { reportId: { $eq: options.reportId } } : undefined;
  const response = await pinecone.index(indexName).namespace(namespace).query({
    vector,
    topK: options.topK ?? 5,
    includeMetadata: true,
    filter,
  });

  const matches = response.matches.filter((match) =>
    match.metadata
    && typeof match.metadata.text === "string"
    && (options.minScore === undefined || (match.score ?? 0) >= options.minScore)
  );
  return matches.length > 0
    ? matches.map((match) => String(match.metadata?.text)).join("\n\n")
    : "<nomatches>";
}

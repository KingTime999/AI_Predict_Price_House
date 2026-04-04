import { GoogleGenAI, Type } from "@google/genai";

const apiKey = process.env.GEMINI_API_KEY || "";
const ai = new GoogleGenAI({ apiKey });

export interface PredictionResult {
  estimatedValue: number;
  confidence: number;
  trend: number;
  analysis: string;
}

export async function predictPropertyPrice(data: {
  area: number;
  location: string;
  propertyType: string;
  bedrooms: number;
  bathrooms: number;
}): Promise<PredictionResult> {
  const model = "gemini-3-flash-preview";
  
  const prompt = `
    As a professional real estate appraiser and data scientist, provide a detailed valuation prediction for the following property:
    - Area: ${data.area} sqft
    - Location: ${data.location}
    - Property Type: ${data.propertyType}
    - Bedrooms: ${data.bedrooms}
    - Bathrooms: ${data.bathrooms}

    Provide the output in JSON format with the following fields:
    - estimatedValue: (number) The predicted market value in USD.
    - confidence: (number) Percentage (0-100) of model confidence based on typical market volatility in that area.
    - trend: (number) Predicted 12-month appreciation percentage (e.g., 5.1).
    - analysis: (string) A brief architectural and market analysis (max 150 words).
  `;

  try {
    const response = await ai.models.generateContent({
      model,
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            estimatedValue: { type: Type.NUMBER },
            confidence: { type: Type.NUMBER },
            trend: { type: Type.NUMBER },
            analysis: { type: Type.STRING },
          },
          required: ["estimatedValue", "confidence", "trend", "analysis"],
        },
      },
    });

    const result = JSON.parse(response.text || "{}");
    return result as PredictionResult;
  } catch (error) {
    console.error("Prediction error:", error);
    throw new Error("Failed to generate prediction. Please try again.");
  }
}

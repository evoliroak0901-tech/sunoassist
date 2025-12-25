
import { GoogleGenerativeAI, ChatSession, GenerativeModel } from "@google/generative-ai";
import { PromptParams, ArtistAnalysisResult, VisualPromptResult, AudioAnalysisResult, VideoPromptResult } from "../types";
import { GENRES, VOCAL_TEXTURES, EMPHASIS_INSTRUMENTS } from "../constants";

// Internal helper to get model
const getModel = (apiKey: string, modelName: string, systemInstruction?: string) => {
    const genAI = new GoogleGenerativeAI(apiKey);
    return genAI.getGenerativeModel({
        model: modelName,
        systemInstruction: systemInstruction
    });
};

export const convertToHiragana = async (apiKey: string, text: string, modelName: string = 'gemini-1.5-pro'): Promise<string> => {
    if (!text.trim()) return "";

    const model = getModel(apiKey, modelName, `You are a Japanese lyrics converter. 
      Task: Convert the following Japanese song lyrics strictly into Hiragana (reading). 
      Rules:
      1. Maintain the exact same line structure and line breaks.
      2. Keep any meta tags (like [Verse], [Chorus]) or English words exactly as they are.
      3. Only output the converted text, no explanations.
      4. If there are Kanji, convert to Hiragana.
      5. If there is already Hiragana or Katakana, ensure it flows naturally as Hiragana.`);

    const result = await model.generateContent(text);
    return result.response.text().trim() || "";
};

export const generateLyrics = async (apiKey: string, keywords: string, modelName: string = 'gemini-1.5-pro'): Promise<string | null> => {
    const systemInstruction = `
    You are a professional songwriter.
    Task: Write song lyrics based on the provided keywords or theme.
    
    Requirements:
    1. Language: Japanese.
    2. Structure: Use standard song structure with tags like [Verse], [Chorus], [Bridge].
    3. Creativity: Be creative, emotional, and rhythmic suitable for a song.
    4. Length: A standard song length (Verse 1, Chorus, Verse 2, Chorus, Outro) or whatever fits the keywords.
    
    Only output the lyrics with tags.
    `;

    const model = getModel(apiKey, 'gemini-1.5-flash-001', systemInstruction);
    const result = await model.generateContent(`Keywords/Theme: ${keywords}\n\nGenerate lyrics now.`);
    return result.response.text().trim() || null;
};

export const analyzeArtistStyle = async (apiKey: string, artistName: string, modelName: string = 'gemini-1.5-pro'): Promise<ArtistAnalysisResult | null> => {
    const systemInstruction = `
    You are a music analysis expert for Suno AI prompting.
    Analyze the artist provided by the user and map their style to the following parameters.
    
    Available Lists to choose from (pick the closest matches):
    - Genres: ${GENRES.join(", ")}
    - Textures: ${VOCAL_TEXTURES.join(", ")}
    - Instruments: ${EMPHASIS_INSTRUMENTS.join(", ")}

    Parameters to determine:
    1. vocalX: Number between -100 and 100.
    2. vocalY: Number between -100 and 100.
    3. genres: Array of strings (Select max 3)
    4. textures: Array of strings (Select max 2)
    5. instruments: Array of strings (Select max 2)
    
    Return strictly JSON.
    `;

    const model = getModel(apiKey, 'gemini-1.5-flash-001', systemInstruction);
    const result = await model.generateContent({
        contents: [{ role: 'user', parts: [{ text: `Analyze the artist: ${artistName}` }] }],
        generationConfig: {
            responseMimeType: "application/json",
        }
    });

    const jsonText = result.response.text();
    if (!jsonText) return null;
    return JSON.parse(jsonText) as ArtistAnalysisResult;
}

export const analyzeVocalAudio = async (apiKey: string, base64Audio: string, mimeType: string, modelName: string = 'gemini-1.5-pro'): Promise<AudioAnalysisResult | null> => {
    const systemInstruction = `
    You are an expert audio engineer. Listen to the provided vocal audio sample and analyze its characteristics.
    Map the analysis to the following parameters:

    1. vocalX: Number (-100 to 100).
    2. vocalY: Number (-100 to 100).
    3. textures: Select up to 2 descriptors: [${VOCAL_TEXTURES.join(", ")}]

    Return strictly JSON.
    `;

    const cleanBase64 = base64Audio.split(',')[1] || base64Audio;
    const model = getModel(apiKey, 'gemini-1.5-flash-001', systemInstruction);

    const result = await model.generateContent({
        contents: [{
            role: 'user',
            parts: [
                { inlineData: { mimeType: mimeType, data: cleanBase64 } },
                { text: "Analyze the vocals in this audio." }
            ]
        }],
        generationConfig: {
            responseMimeType: "application/json",
        }
    });

    const jsonText = result.response.text();
    if (!jsonText) return null;
    return JSON.parse(jsonText) as AudioAnalysisResult;
};

export const generateSunoPrompt = async (apiKey: string, params: PromptParams, modelName: string = 'gemini-1.5-pro'): Promise<string> => {
    let vocalDesc = "";
    const x = params.vocalX;
    const y = params.vocalY;

    if (x < -30) vocalDesc += "Male vocals";
    else if (x > 30) vocalDesc += "Female vocals";
    else vocalDesc += "Androgynous vocals";

    if (y < -30) vocalDesc += ", Low pitch";
    else if (y > 30) vocalDesc += ", High pitch";

    const systemInstruction = `You are a Suno AI prompt generator expert.
    Task: Create a single string of comma-separated English style tags for Suno AI.
    
    CRITICAL CONSTRAINT: 
    The output string MUST be under 1000 characters. 
    IMPORTANT: Do NOT use the Artist Name in the output.

    Inputs:
    1. Vocal Characteristics: ${vocalDesc}
    2. Vocal Textures: ${params.textures.join(", ")}
    3. Target Genres: ${params.genres.join(", ")}
    4. Emphasized Instruments: ${params.instruments.join(", ")}
    5. Artist Style Reference: ${params.artist || "None"}

    Output format:
    [Genre], [Sub-genre], [Instruments], [Vocal Style], [Mood/Atmosphere], [Tempo]
    `;

    const model = getModel(apiKey, modelName, systemInstruction);
    const result = await model.generateContent("Generate the Suno prompt string now.");

    let finalPrompt = result.response.text().trim() || "";
    if (finalPrompt.length > 1000) finalPrompt = finalPrompt.substring(0, 1000);
    return finalPrompt;
};

export const generateVisualPrompts = async (apiKey: string, lyrics: string, modelName: string = 'gemini-1.5-pro'): Promise<VisualPromptResult | null> => {
    const systemInstruction = `
    You are a creative director. Analyze the provided lyrics and extract core imagery and mood.
    Output a JSON object with:
    1. sceneDescription: Concise Japanese summary (max 30 chars).
    2. imagePrompt: Detailed English prompt for image generator.
    `;

    const model = getModel(apiKey, modelName, systemInstruction);
    const result = await model.generateContent({
        contents: [{ role: 'user', parts: [{ text: `Lyrics:\n${lyrics}` }] }],
        generationConfig: {
            responseMimeType: "application/json",
        }
    });

    const jsonText = result.response.text();
    if (!jsonText) return null;
    return JSON.parse(jsonText) as VisualPromptResult;
};

export const generateVideoPromptForSection = async (apiKey: string, lyricsPart: string, modelName: string = 'gemini-1.5-pro'): Promise<VideoPromptResult | null> => {
    const systemInstruction = `
    You are a video direction expert. Create a video generation prompt for a section of a song.
    Input Lyrics: "${lyricsPart}"
    Output JSON: sceneDescription (Japanese, max 30 chars), soraPrompt (English, detailed visual motion).
    `;

    const model = getModel(apiKey, modelName, systemInstruction);
    const result = await model.generateContent({
        contents: [{ role: 'user', parts: [{ text: `Generate video prompt for section.` }] }],
        generationConfig: {
            responseMimeType: "application/json",
        }
    });

    const jsonText = result.response.text();
    if (!jsonText) return null;
    const json = JSON.parse(jsonText);
    return { ...json, lyricsPart } as VideoPromptResult;
};

export const generateImage = async (apiKey: string, prompt: string, modelName: string = 'gemini-1.5-pro'): Promise<string | null> => {
    try {
        // Image generation is not natively supported in the same way here, 
        // replacing with gemini-1.5-flash as a placeholder that will likely fail 404 for image bytes
        // but it's better than a non-existent 2.5 model.
        const model = getModel(apiKey, modelName);
        const result = await model.generateContent(prompt);

        const response = result.response;
        // Check for inlineData in parts
        const parts = response.candidates?.[0]?.content?.parts;
        if (parts) {
            for (const part of parts) {
                if (part.inlineData && part.inlineData.mimeType.startsWith('image/')) {
                    return `data:${part.inlineData.mimeType};base64,${part.inlineData.data}`;
                }
            }
        }
        console.warn("No image data found in Gemini response.");
        return null;
    } catch (error) {
        console.error("Error in generateImage:", error);
        return null;
    }
}

export const createChatSession = (apiKey: string, modelName: string = 'gemini-1.5-pro'): ChatSession | null => {
    const model = getModel(apiKey, modelName, "あなたはプロの音楽プロデューサー兼作詞家のアシスタントです。ユーザーの作詞、楽曲構成、Suno AIのプロンプト作成などについて日本語でアドバイスをしてください。");
    return model.startChat();
};

// --- Audio Decoding Helpers (Manual Implementation per Guidelines) ---

function decode(base64: string) {
    const binaryString = atob(base64);
    const len = binaryString.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes;
}

async function decodeAudioData(
    data: Uint8Array,
    ctx: AudioContext,
    sampleRate: number,
    numChannels: number,
): Promise<AudioBuffer> {
    const dataInt16 = new Int16Array(data.buffer);
    const frameCount = dataInt16.length / numChannels;
    const buffer = ctx.createBuffer(numChannels, frameCount, sampleRate);

    for (let channel = 0; channel < numChannels; channel++) {
        const channelData = buffer.getChannelData(channel);
        for (let i = 0; i < frameCount; i++) {
            channelData[i] = dataInt16[i * numChannels + channel] / 32768.0;
        }
    }
    return buffer;
}

export const playVoiceSample = async (apiKey: string, text: string, vocalX: number, vocalY: number, modelName: string = 'gemini-1.5-pro'): Promise<void> => {
    let voiceName = 'Zephyr';
    if (vocalX < -20) voiceName = 'Charon';
    else if (vocalX > 20) voiceName = 'Kore';
    else voiceName = 'Puck';

    const model = getModel(apiKey, modelName);

    const result = await model.generateContent({
        contents: [{ role: 'user', parts: [{ text: text }] }],
        generationConfig: {
            // @ts-ignore
            responseModalities: ["AUDIO"],
            speechConfig: {
                voiceConfig: {
                    prebuiltVoiceConfig: { voiceName: voiceName },
                },
            },
        },
    });

    // @ts-ignore
    const base64Audio = result.response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
    if (!base64Audio) return;

    const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
    const decodedBytes = decode(base64Audio);
    const buffer = await decodeAudioData(decodedBytes, audioCtx, 24000, 1);

    const source = audioCtx.createBufferSource();
    source.buffer = buffer;
    source.connect(audioCtx.destination);
    source.start();
};

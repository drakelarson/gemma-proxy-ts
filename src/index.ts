/**
 * Gemini/Gemma API Proxy - TypeScript/Hono
 * OpenAI-compatible interface for Google Gemini models
 * 
 * Deployable to: Vercel Edge, Bun, Cloudflare Workers
 * 
 * Set GEMINI_API_KEY in Vercel Environment Variables
 */

import { Hono } from 'hono'
import { cors } from 'hono/cors'

const BASE_URL = 'https://generativelanguage.googleapis.com/v1beta'

// API key from environment variable
// Set GEMINI_API_KEY in Vercel Environment Variables
const API_KEY = process.env.GEMINI_API_KEY

if (!API_KEY) {
  console.error('[GEMINI-PROXY] ERROR: No API key! Set GEMINI_API_KEY env var.')
}

// Model name mapping (OpenAI-style to Gemini)
// Use stable model names for production
const MODEL_MAP: Record<string, string> = {
  'gemini-2.5-pro': 'gemini-2.5-pro',
  'gemini-2.5-flash': 'gemini-2.5-flash',
  'gemini-2.5-flash-lite': 'gemini-2.5-flash-lite',
  'gemini-3-flash-preview': 'gemini-3-flash-preview',
  'gemini-3.1-pro-preview': 'gemini-3.1-pro-preview',
  'gemini-3.1-flash-lite-preview': 'gemini-3.1-flash-lite-preview',
  // Legacy aliases
  'gemini-1.5-pro': 'gemini-1.5-pro',
  'gemini-1.5-flash': 'gemini-1.5-flash',
  'gemini-2.0-flash': 'gemini-2.0-flash',
}

let totalRequests = 0
let totalErrors = 0

const app = new Hono()

app.use('*', cors())

app.get('/', (c) => {
  return c.json({
    status: 'ok',
    provider: 'gemini',
    hasKey: !!API_KEY,
    models: Object.keys(MODEL_MAP),
    stats: { totalRequests, totalErrors }
  })
})

app.get('/v1/models', (c) => {
  return c.json({
    object: 'list',
    data: Object.entries(MODEL_MAP).map(([openaiName, geminiName]) => ({
      id: openaiName,
      object: 'model',
      created: Date.now(),
      owned_by: 'google',
      gemini_model: geminiName
    }))
  })
})

// Convert OpenAI messages to Gemini format
function convertMessages(openaiMessages: any[]): { contents: any[], systemInstruction?: any } {
  const contents: any[] = []
  let systemInstruction: any = undefined
  
  for (const msg of openaiMessages) {
    const role = msg.role
    
    if (role === 'system') {
      systemInstruction = { parts: [{ text: msg.content }] }
      continue
    }
    
    // Map roles: assistant -> model, user -> user
    const geminiRole = role === 'assistant' ? 'model' : 'user'
    
    // Handle content (string or array)
    const parts: any[] = []
    if (typeof msg.content === 'string') {
      if (msg.content) parts.push({ text: msg.content })
    } else if (Array.isArray(msg.content)) {
      // Handle multimodal content (text + images)
      for (const part of msg.content) {
        if (part.type === 'text') {
          parts.push({ text: part.text })
        } else if (part.type === 'image_url' && part.image_url?.url) {
          // Extract base64 data from data URL
          const url = part.image_url.url
          if (url.startsWith('data:')) {
            const [header, base64] = url.split(',')
            const mimeType = header.match(/data:([^;]+)/)?.[1] || 'image/jpeg'
            parts.push({
              inlineData: {
                mimeType,
                data: base64
              }
            })
          }
        }
      }
    }
    
    if (parts.length > 0) {
      contents.push({ role: geminiRole, parts })
    }
  }
  
  return { contents, systemInstruction }
}

// Convert Gemini response to OpenAI format
function convertResponse(geminiResp: any, model: string, stream: boolean): any {
  const candidates = geminiResp.candidates || []
  const candidate = candidates[0] || {}
  const content = candidate.content || {}
  const parts = content.parts || []
  
  // Extract text from parts
  let text = ''
  for (const part of parts) {
    if (part.text) text += part.text
  }
  
  const finishReason = candidate.finishReason === 'STOP' ? 'stop' : candidate.finishReason?.toLowerCase() || 'stop'
  
  if (stream) {
    // Streaming format
    return {
      id: `chatcmpl-${Date.now()}`,
      object: 'chat.completion.chunk',
      created: Math.floor(Date.now() / 1000),
      model,
      choices: [{
        index: 0,
        delta: { content: text },
        finish_reason: null
      }]
    }
  } else {
    // Non-streaming format
    return {
      id: `chatcmpl-${Date.now()}`,
      object: 'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model,
      choices: [{
        index: 0,
        message: { role: 'assistant', content: text },
        finish_reason: finishReason
      }],
      usage: {
        prompt_tokens: geminiResp.usageMetadata?.promptTokenCount || 0,
        completion_tokens: geminiResp.usageMetadata?.candidatesTokenCount || 0,
        total_tokens: geminiResp.usageMetadata?.totalTokenCount || 0
      }
    }
  }
}

app.post('/v1/chat/completions', async (c) => {
  totalRequests++
  const startTime = Date.now()
  
  if (!API_KEY) {
    return c.json({ error: 'GEMINI_API_KEY not configured' }, 500)
  }
  
  try {
    const body = await c.req.json()
    const { model: requestedModel, messages, stream = false, ...rest } = body
    
    // Map model name
    const geminiModel = MODEL_MAP[requestedModel] || requestedModel
    
    // Convert messages
    const { contents, systemInstruction } = convertMessages(messages)
    
    // Build Gemini request
    const geminiRequest: any = {
      contents,
      generationConfig: {}
    }
    
    if (systemInstruction) {
      geminiRequest.systemInstruction = systemInstruction
    }
    
    // Map OpenAI params to Gemini params
    if (rest.temperature !== undefined) {
      geminiRequest.generationConfig.temperature = rest.temperature
    }
    if (rest.top_p !== undefined) {
      geminiRequest.generationConfig.topP = rest.top_p
    }
    if (rest.max_tokens !== undefined) {
      geminiRequest.generationConfig.maxOutputTokens = rest.max_tokens
    }
    if (rest.stop !== undefined) {
      geminiRequest.generationConfig.stopSequences = Array.isArray(rest.stop) ? rest.stop : [rest.stop]
    }
    
    console.log(`[GEMINI-PROXY] POST /v1/chat/completions → model=${geminiModel}, stream=${stream}`)
    
    // Choose endpoint based on stream flag
    const endpoint = stream ? 'streamGenerateContent' : 'generateContent'
    const url = `${BASE_URL}/models/${geminiModel}:${endpoint}?key=${API_KEY}`
    
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(geminiRequest)
    })
    
    if (!response.ok) {
      const errorText = await response.text()
      console.error(`[GEMINI-PROXY] Error ${response.status}: ${errorText}`)
      totalErrors++
      return c.json({ 
        error: `Gemini API error: ${response.status}`,
        details: errorText
      }, response.status as 400 | 401 | 403 | 404 | 429 | 500 | 502 | 503)
    }
    
    if (stream) {
      // Gemini returns JSON array for streaming, not SSE
      // Read full response and convert to SSE
      const text = await response.text()
      
      // Parse the JSON array (Gemini returns array of responses)
      let geminiResponses: any[]
      try {
        // Handle both array and single object responses
        const parsed = JSON.parse(text)
        geminiResponses = Array.isArray(parsed) ? parsed : [parsed]
      } catch (e) {
        console.error(`[GEMINI-PROXY] Parse error: ${e}`)
        return c.json({ error: 'Failed to parse Gemini response' }, 502)
      }
      
      // Create SSE stream
      const encoder = new TextEncoder()
      let fullText = ''
      
      const stream = new ReadableStream({
        start(controller) {
          for (const geminiResp of geminiResponses) {
            const parts = geminiResp.candidates?.[0]?.content?.parts || []
            for (const part of parts) {
              if (part.text) {
                fullText += part.text
                const chunk = convertResponse(geminiResp, requestedModel, true)
                chunk.choices[0].delta.content = part.text
                controller.enqueue(encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`))
              }
            }
          }
          
          // Send final chunk
          const finalChunk = {
            id: `chatcmpl-${Date.now()}`,
            object: 'chat.completion.chunk',
            created: Math.floor(Date.now() / 1000),
            model: requestedModel,
            choices: [{ index: 0, delta: {}, finish_reason: 'stop' }]
          }
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(finalChunk)}\n\n`))
          controller.enqueue(encoder.encode('data: [DONE]\n\n'))
          controller.close()
          
          console.log(`[GEMINI-PROXY] ← ${Date.now() - startTime}ms, ${fullText.length} chars`)
        }
      })
      
      return new Response(stream, {
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          'Connection': 'keep-alive',
          'X-Accel-Buffering': 'no',
        }
      })
    } else {
      // Non-streaming response
      const geminiResp = await response.json()
      const openaiResp = convertResponse(geminiResp, requestedModel, false)
      
      console.log(`[GEMINI-PROXY] ← ${Date.now() - startTime}ms`)
      return c.json(openaiResp)
    }
    
  } catch (error) {
    console.error(`[GEMINI-PROXY] ERROR: ${error}`)
    totalErrors++
    return c.json({ error: String(error) }, 502)
  }
})

// Export for Vercel Edge
export default app

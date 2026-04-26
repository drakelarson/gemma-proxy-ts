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
// Set GEMINI_API_KEY in Vercel pEnvironment Variables
const API_KEY = process.env.GEMINI_API_KEY

if (!API_KEY) {
  console.error('[GEMINI-PROXY] ERROR: No API key! Set GEMINI_API_KEY env var.')
}

// Model name mapping (OpenAI-style to Gemini API names)
// Use stable model names for production
const MODEL_MAP: Record<string, string> = {
  // Gemini models
  'gemini-2.5-pro': 'gemini-2.5-pro',
  'gemini-2.5-flash': 'gemini-2.5-flash',
  'gemini-2.5-flash-lite': 'gemini-2.5-flash-lite',
  'gemini-3-flash-preview': 'gemini-3-flash-preview',
  'gemini-3.1-pro-preview': 'gemini-3.1-pro-preview',
  'gemini-3.1-flash-lite-preview': 'gemini-3.1-flash-lite-preview',
  // Legacy Gemini
  'gemini-1.5-pro': 'gemini-1.5-pro',
  'gemini-1.5-flash': 'gemini-1.5-flash',
  'gemini-2.0-flash': 'gemini-2.0-flash',
  // Gemma 4 models
  'gemma-4-31b-it': 'gemma-4-31b-it',
  'gemma-4-26b-a4b-it': 'gemma-4-26b-a4b-it',
  // Gemma 3
  'gemma-3-27b-it': 'gemma-3-27b-it',
}

const DEFAULT_MODEL = 'gemini-2.5-flash'

// Heartbeat for keep-alive during streaming
const HEARTBEAT_INTERVAL_MS = 2000
const HEARTBEAT_BYTE = ': keep-alive\n\n'

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
  
  // Build a map of tool_call_id -> function name for tool responses
  const toolCallIdToName: Map<string, string> = new Map()
  for (const msg of openaiMessages) {
    if (msg.role === 'assistant' && msg.tool_calls) {
      for (const tc of msg.tool_calls) {
        if (tc.id && tc.function?.name) {
          toolCallIdToName.set(tc.id, tc.function.name)
        }
      }
    }
  }
  
  for (const msg of openaiMessages) {
    const role = msg.role
    
    if (role === 'system') {
      systemInstruction = { parts: [{ text: msg.content }] }
      continue
    }
    
    // Map roles: assistant -> model, user -> user, tool -> user (with functionResponse)
    let geminiRole = 'user'
    if (role === 'assistant') geminiRole = 'model'
    else if (role === 'tool') geminiRole = 'user'
    
    // Handle content (string or array)
    const parts: any[] = []
    
    // Handle tool calls in assistant message
    if (role === 'assistant' && msg.tool_calls) {
      // EXCEPTION: Keep reasoning_content for tool calls per Gemma docs
      if (msg.reasoning_content) {
        parts.push({ thought: true, text: msg.reasoning_content })
      }
      for (const tc of msg.tool_calls) {
        parts.push({
          functionCall: {
            name: tc.function.name,
            args: (() => {
              try {
                return JSON.parse(tc.function.arguments || '{}')
              } catch {
                const argStr = tc.function.arguments
                const matches = argStr.match(/\\{[^{}]*\\}/g)
                if (matches && matches.length > 0) {
                  try {
                    return JSON.parse(matches[0])
                  } catch {
                    return {}
                  }
                }
                return {}
              }
            })(),
          }
        })
      }
      // Add content if present
      if (msg.content && typeof msg.content === 'string' && msg.content.trim()) {
        parts.push({ text: msg.content })
      }
    }
    // Handle assistant message without tool calls (strip reasoning_content)
    else if (role === 'assistant') {
      // Standard multi-turn: strip reasoning_content - Gemini generates thoughts fresh each turn
      if (msg.content && typeof msg.content === 'string' && msg.content.trim()) {
        parts.push({ text: msg.content })
      }
    }
    // Handle tool response
    else if (role === 'tool') {
      // Get function name from our lookup map
      const functionName = msg.name || toolCallIdToName.get(msg.tool_call_id) || 'unknown_function'
      
      // Parse tool response content
      let responseContent: any
      if (typeof msg.content === 'string') {
        try {
          responseContent = JSON.parse(msg.content || '{}')
        } catch {
          // Non-JSON content - wrap as content string
          responseContent = { content: msg.content }
        }
      } else {
        responseContent = msg.content || {}
      }
      
      // Gemini's response field MUST be an object, not an array
      // If responseContent is an array, wrap it in an object
      if (Array.isArray(responseContent)) {
        responseContent = { result: responseContent }
      }
      // If responseContent is a primitive (number, string, boolean), wrap in object
      if (typeof responseContent !== 'object' || responseContent === null) {
        responseContent = { result: responseContent }
      }
      
      parts.push({
        functionResponse: {
          name: functionName,
          response: responseContent
        }
      })
    }
    else if (typeof msg.content === 'string') {
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

// Strip OpenAI-specific fields from JSON schema that Gemini doesn't support
function stripOpenAIFields(schema: any): any {
  if (!schema || typeof schema !== 'object') return schema
  if (Array.isArray(schema)) return schema.map(stripOpenAIFields)
  
  const result: any = {}
  for (const [key, value] of Object.entries(schema)) {
    // Skip OpenAI-specific fields that Gemini doesn't support
    if (key === 'additionalProperties' || key === '$schema' || key === 'strict') continue
    
    // Filter out empty strings from enum arrays (Gemini rejects them)
    if (key === 'enum' && Array.isArray(value)) {
      const filtered = value.filter((v: any) => v !== '')
      if (filtered.length > 0) {
        result[key] = filtered
      }
      continue
    }
    
    result[key] = stripOpenAIFields(value)
  }
  return result
}

// Convert OpenAI tools to Gemini function declarations
function convertTools(openaiTools: any[]): any[] {
  if (!openaiTools || openaiTools.length === 0) return []
  
  const functionDeclarations = openaiTools.map(tool => {
    if (tool.type === 'function') {
      const fn = tool.function
      return {
        name: fn.name,
        description: fn.description || '',
        parameters: stripOpenAIFields(fn.parameters) || { type: 'object', properties: {} }
      }
    }
    return null
  }).filter(Boolean)
  
  return functionDeclarations
}

// Convert Gemini response to OpenAI format
function convertResponse(geminiResp: any, model: string, stream: boolean): any {
  const candidates = geminiResp.candidates || []
  const candidate = candidates[0] || {}
  const content = candidate.content || {}
  const parts = content.parts || []
  
  // Separate thoughts from actual content (Gemma 4 uses thought: true)
  let text = ''
  let thoughts = ''
  const toolCalls: any[] = []
  let toolCallIdx = 0  // Use counter for unique IDs
  
  for (const part of parts) {
    if (part.thought) {
      // This is a thinking/reasoning part - skip for main content
      if (part.text) thoughts += part.text
    } else if (part.functionCall) {
      // Function call - convert to OpenAI tool_calls format
      // Use counter + timestamp + random for truly unique IDs
      const tc: any = {
        id: part.functionCall.id || `call_${Date.now()}_${toolCallIdx}_${Math.random().toString(36).substr(2, 9)}`,
        type: 'function',
        function: {
          name: part.functionCall.name,
          arguments: JSON.stringify(part.functionCall.args || {})
        }
      }

      toolCalls.push(tc)
      toolCallIdx++
    } else if (part.text && part.text.trim()) {
      text += part.text
    }
  }
  
  const finishReason = candidate.finishReason === 'STOP' ? 'stop' : 
    (toolCalls.length > 0 ? 'tool_calls' : candidate.finishReason?.toLowerCase() || 'stop')
  
  if (stream) {
    // Streaming format - only return non-thought content
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
    // Non-streaming format - include thoughts in reasoning_content if present
    const message: any = { role: 'assistant', content: text || "" }
    if (thoughts) {
      message.reasoning_content = thoughts
    }
    if (toolCalls.length > 0) {
      message.tool_calls = toolCalls
    }
    
    return {
      id: `chatcmpl-${Date.now()}`,
      object: 'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model,
      choices: [{
        index: 0,
        message,
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
    const geminiModel = MODEL_MAP[requestedModel] || DEFAULT_MODEL
    
    // Convert messages
    const { contents, systemInstruction } = convertMessages(messages)
    
    // Convert tools if provided
    const tools = rest.tools ? convertTools(rest.tools) : []
    
    // Build Gemini request
    const geminiRequest: any = {
      contents,
      generationConfig: {
        thinkingConfig: {
          thinkingLevel: "high"
        }
      }
    }
    
    if (systemInstruction) {
      geminiRequest.systemInstruction = systemInstruction
    }
    
    // Add tools if provided
    if (tools.length > 0) {
      geminiRequest.tools = [{ functionDeclarations: tools }]
    }
    
    // Map OpenAI params to Gemini params
    // Hardcode temperature to 1.2 - do not allow override
    geminiRequest.generationConfig.temperature = 1.2
    if (rest.top_p !== undefined) {
      geminiRequest.generationConfig.topP = rest.top_p
    }
    if (rest.top_k !== undefined) {
      geminiRequest.generationConfig.topK = rest.top_k
    }
    if (rest.max_tokens !== undefined) {
      geminiRequest.generationConfig.maxOutputTokens = rest.max_tokens
    }
    if (rest.stop !== undefined) {
      geminiRequest.generationConfig.stopSequences = Array.isArray(rest.stop) ? rest.stop.filter(Boolean) : [rest.stop]
    }
    
    console.log(`[GEMINI-PROXY] POST /v1/chat/completions → model=${geminiModel}, stream=${stream}`)
    
    // Choose endpoint based on stream flag
    // For streaming, use alt=sse to get SSE format from Gemini
    const endpoint = stream 
      ? `streamGenerateContent?alt=sse&key=${API_KEY}` 
      : `generateContent?key=${API_KEY}`
    const url = `${BASE_URL}/models/${geminiModel}:${endpoint}`
    
    console.log(`[GEMINI-PROXY] → ${url.replace(API_KEY!, '[REDACTED]')}`)
    
    // Retry logic for 500 errors
    const MAX_RETRIES = 3
    const RETRY_DELAY_MS = 1000
    let response: Response | null = null
    
    for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(geminiRequest)
      })
      
      if (res.ok) {
        response = res
        break
      }
      
      const errorText = await res.text()
      
      // Only retry on 500 (internal server error)
      if (res.status === 500 && attempt < MAX_RETRIES - 1) {
        console.log(`[GEMINI-PROXY] 500 error, retry ${attempt + 1}/${MAX_RETRIES} after ${RETRY_DELAY_MS * (attempt + 1)}ms`)
        await new Promise(r => setTimeout(r, RETRY_DELAY_MS * (attempt + 1)))
        continue
      }
      
      // Non-500 error or final attempt failed - return error immediately
      console.error(`[GEMINI-PROXY] Error ${res.status}: ${errorText}`)
      totalErrors++
      return c.json({ 
        error: `Gemini API error: ${res.status}`,
        details: errorText
      }, res.status as 400 | 401 | 403 | 404 | 429 | 500 | 502 | 503)
    }
    
    // Should never happen, but TypeScript needs this check
    if (!response) {
      return c.json({ error: 'No response from Gemini API' }, 502)
    }
    
    if (stream) {
      // Real streaming - Gemini returns SSE with alt=sse
      // Read SSE stream and convert each Gemini event to OpenAI format
      const reader = response.body?.getReader()
      if (!reader) {
        return c.json({ error: 'No response body' }, 502)
      }
      
      const encoder = new TextEncoder()
      const decoder = new TextDecoder()
      let buffer = ''
      let hadToolCall = false
      let thoughtBuffer: string[] = []
      let toolCallIndex = 0  // Track tool call count for proper delta indexing
      
      return new Response(
        new ReadableStream({
          async start(controller) {
            // Track last activity time for keep-alive heartbeats
            let lastActivity = Date.now()
            let heartbeatTimer: any = null
            let fullContent = ''
            
            // Start heartbeat to prevent connection drops
            const startHeartbeat = () => {
              heartbeatTimer = setInterval(() => {
                const now = Date.now()
                // Send empty data event to keep TCP alive without breaking OpenAI parsers
                if (now - lastActivity >= HEARTBEAT_INTERVAL_MS) {
                  controller.enqueue(encoder.encode('data: \n\n'))
                  lastActivity = now
                }
              }, HEARTBEAT_INTERVAL_MS)
            }
            
            const stopHeartbeat = () => {
              if (heartbeatTimer) {
                clearInterval(heartbeatTimer)
                heartbeatTimer = null
              }
            }
            
            startHeartbeat()
            
            try {
              while (true) {
                const { done, value } = await reader.read()
                if (done) {
                  stopHeartbeat()
                  const finalFinishReason = hadToolCall ? 'tool_calls' : 'stop'
                  
                  // Construct a minimal "done" chunk. 
                  // Some clients fail if delta is empty.
                  const doneChunk = {
                    id: `chatcmpl-${Date.now()}`,
                    object: 'chat.completion.chunk',
                    created: Math.floor(Date.now()/1000),
                    model: requestedModel,
                    choices: [{
                      index: 0,
                      delta: {}, 
                      finish_reason: finalFinishReason
                    }]
                  }
                  controller.enqueue(encoder.encode(`data: ${JSON.stringify(doneChunk)}\n\n`))
                  controller.enqueue(encoder.encode('data: [DONE]\n\n'))
                  break
                }
                
                lastActivity = Date.now()
                buffer += decoder.decode(value, { stream: true })
                
                const lines = buffer.split('\n')
                buffer = lines.pop() || ''
                
                for (const line of lines) {
                  if (line.startsWith('data: ')) {
                    const data = line.slice(6).trim()
                    if (!data || data === '[DONE]') continue
                    
                    // DEBUG: Log raw data string for complete visibility
                    console.error(`[GEMINI-PROXY] DEBUG: RAW DATA: ${data}`)
                    
                    lastActivity = Date.now()
                    
                    try {
                      const geminiChunk = JSON.parse(data)
                      const parts = geminiChunk.candidates?.[0]?.content?.parts || []
                      
                      for (const part of parts) {
                        // FORCE log to error to ensure capture in Vercel logs
                        if (part.text && part.text.trim()) {
                          console.error(`[GEMINI-PROXY] DEBUG: Processing part: ${JSON.stringify(part)}`)
                        }
                        
                        if (part.functionCall) {
                          hadToolCall = true
                          console.error(`[GEMINI-PROXY] DEBUG: Processing functionCall: ${JSON.stringify(part.functionCall)}`)
                          const toolCallChunk = {
                            id: `chatcmpl-${Date.now()}`,
                            object: 'chat.completion.chunk',
                            created: Math.floor(Date.now() / 1000),
                            model: requestedModel,
                            choices: [{
                              index: 0,
                              delta: {
                                tool_calls: [{
                                  index: toolCallIndex,
                                  id: part.functionCall.id || `call_${Date.now()}_${toolCallIndex}_${Math.random().toString(36).substr(2, 9)}`,
                                  type: 'function',
                                  function: {
                                    name: part.functionCall.name,
                                    arguments: JSON.stringify(part.functionCall.args || {})
                                  }
                                }]
                              },
                              finish_reason: null
                            }]
                          }
                          toolCallIndex++
                          console.error(`[GEMINI-PROXY] DEBUG: Enqueueing tool call chunk: ${JSON.stringify(toolCallChunk)}`)
                          controller.enqueue(encoder.encode(`data: ${JSON.stringify(toolCallChunk)}\n\n`))
                        } else if (part.thought && part.text) {
                          // Send thought as delta.reasoning_content for UI recognition
                          const thoughtChunk = {
                            id: `chatcmpl-${Date.now()}`,
                            object: 'chat.completion.chunk',
                            created: Math.floor(Date.now() / 1000),
                            model: requestedModel,
                            choices: [{
                              index: 0,
                              delta: { reasoning_content: part.text },
                              finish_reason: null
                            }]
                          }
                          controller.enqueue(encoder.encode(`data: ${JSON.stringify(thoughtChunk)}\n\n`))
                          continue
                        } else if (part.text && part.text.trim()) {
                          // Stream content chunks immediately instead
                          const contentChunk = {
                            id: `chatcmpl-${Date.now()}`,
                            object: 'chat.completion.chunk',
                            created: Math.floor(Date.now() / 1000),
                            model: requestedModel,
                            choices: [{
                              index: 0,
                              delta: { content: part.text },
                              finish_reason: null
                            }]
                          }
                          controller.enqueue(new TextEncoder().encode(`data: ${JSON.stringify(contentChunk)}\n\n`))
                        }
                      }
                    } catch (e) {
                      console.error('[GEMINI-PROXY] Parse error for data:', data.substring(0, 100))
                    }
                  }
                }
              }
              console.log(`[GEMINI-PROXY] ← ${Date.now() - startTime}ms (streamed)`)
            } catch (e) {
              console.error('[GEMINI-PROXY] Stream error:', e)
              stopHeartbeat()
              controller.close()
            }
          }
        }),
        {
          headers: {
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no',
          }
        }
      )
    } else {
      // Non-streaming response
      const geminiResp = await response.json()
      
      // DEBUG: Log the full Gemini response
      console.log(`[GEMINI-PROXY] DEBUG: Gemini Response: ${JSON.stringify(geminiResp).substring(0, 500)}`)
      
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

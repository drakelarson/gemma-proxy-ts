/**
 * OpenAI-compatible Gemini (Gemma) proxy - stable version
 */

import { Hono } from 'hono'
import { cors } from 'hono/cors'

const app = new Hono()
app.use('*', cors())

const BASE_URL = 'https://generativelanguage.googleapis.com/v1beta'
const API_KEY = process.env.GEMINI_API_KEY!

if (!API_KEY) {
  console.error('Missing GEMINI_API_KEY')
}

/* -----------------------------
 * Model mapping
 * ----------------------------*/
const MODEL_MAP: Record<string, string> = {
  'gemma-4': 'gemma-4-31b-it',
  'gemma-3': 'gemma-3-27b-it',
}

const DEFAULT_MODEL = 'gemma-4-31b-it'

/* -----------------------------
 * Helpers: OpenAI -> Gemini
 * ----------------------------*/
function convertMessages(messages: any[]) {
  const contents: any[] = []
  let systemInstruction = ''

  for (const msg of messages || []) {
    if (msg.role === 'system') {
      systemInstruction += msg.content + '\n'
      continue
    }

    if (msg.role === 'assistant' || msg.role === 'user') {
      contents.push({
        role: msg.role === 'assistant' ? 'model' : 'user',
        parts: [{ text: msg.content }],
      })
    }
  }

  return { contents, systemInstruction: systemInstruction.trim() || undefined }
}

/* -----------------------------
 * Helpers: tools conversion
 * ----------------------------*/
function convertTools(tools: any[]) {
  if (!tools?.length) return []

  return tools.map((t) => ({
    name: t.function?.name,
    description: t.function?.description || '',
    parameters: t.function?.parameters || {},
  }))
}

/* -----------------------------
 * Helpers: Gemini -> OpenAI
 * ----------------------------*/
function convertResponse(gemini: any, model: string) {
  const text =
    gemini?.candidates?.[0]?.content?.parts
      ?.map((p: any) => p.text || '')
      .join('') || ''

  return {
    id: `chatcmpl-${Date.now()}`,
    object: 'chat.completion',
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [
      {
        index: 0,
        message: {
          role: 'assistant',
          content: text,
        },
        finish_reason: 'stop',
      },
    ],
  }
}

/* -----------------------------
 * Streaming endpoint
 * ----------------------------*/
app.post('/v1/chat/completions', async (c) => {
  try {
    const body = await c.req.json()
    const {
      model,
      messages,
      stream = false,
      tools,
    } = body

    const geminiModel = MODEL_MAP[model] || DEFAULT_MODEL

    const { contents, systemInstruction } = convertMessages(messages)
    const functionDeclarations = convertTools(tools)

    const requestBody: any = {
      contents,
      generationConfig: {
        temperature: 1,
        topP: 0.9,
      },
    }

    if (systemInstruction) {
      requestBody.systemInstruction = {
        parts: [{ text: systemInstruction }],
      }
    }

    if (functionDeclarations.length) {
      requestBody.tools = [
        { functionDeclarations },
      ]
    }

    const url = `${BASE_URL}/models/${geminiModel}:${
      stream ? 'streamGenerateContent?alt=sse&key=' : 'generateContent?key='
    }${API_KEY}`

    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestBody),
    })

    /* -----------------------------
     * Non-streaming
     * ----------------------------*/
    if (!stream) {
      const json = await res.json()
      return c.json(convertResponse(json, model))
    }

    /* -----------------------------
     * Streaming (SSE safe)
     * ----------------------------*/
    const reader = res.body?.getReader()
    if (!reader) return c.json({ error: 'No stream' }, 500)

    const decoder = new TextDecoder()

    let buffer = ''

    return new Response(
      new ReadableStream({
        async start(controller) {
          const send = (data: any) => {
            controller.enqueue(
              new TextEncoder().encode(
                `data: ${JSON.stringify(data)}\n\n`
              )
            )
          }

          send({
            id: `chatcmpl-${Date.now()}`,
            object: 'chat.completion.chunk',
            choices: [{ delta: { role: 'assistant' }, index: 0 }],
          })

          while (true) {
            const { done, value } = await reader.read()
            if (done) break

            buffer += decoder.decode(value, { stream: true })

            const lines = buffer.split('\n')
            buffer = lines.pop() || ''

            for (const line of lines) {
              if (!line.startsWith('data:')) continue

              const data = line.replace('data:', '').trim()
              if (!data || data === '[DONE]') continue

              try {
                const json = JSON.parse(data)
                const part =
                  json?.candidates?.[0]?.content?.parts?.[0]

                if (part?.text) {
                  send({
                    id: `chatcmpl-${Date.now()}`,
                    object: 'chat.completion.chunk',
                    choices: [
                      {
                        index: 0,
                        delta: { content: part.text },
                      },
                    ],
                  })
                }
              } catch {
                // ignore malformed chunk
              }
            }
          }

          send({
            id: `chatcmpl-${Date.now()}`,
            object: 'chat.completion.chunk',
            choices: [{ delta: {}, finish_reason: 'stop' }],
          })

          controller.close()
        },
      }),
      {
        headers: {
          'Content-Type': 'text/event-stream',
        },
      }
    )
  } catch (err: any) {
    return c.json({ error: err.message }, 500)
  }
})

export default app

// Pointing to the deployed HuggingFace Space
const API_BASE = "https://viraj0112-gemma4-finetuned.hf.space/api/v1";

interface BackendMessage {
    role: "user" | "assistant";
    content: string; // backend schema uses "content", store uses "text"
}


export const chatWithAIStream = async (
    conversationHistory: { role: "user" | "assistant"; text: string }[],
    onChunk: (chunk: string) => void
): Promise<void> => {
    // Map store shape { role, text } → backend shape { role, content }
    const messages: BackendMessage[] = conversationHistory.map((m) => ({
        role: m.role,
        content: m.text,
    }));

    return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open("POST", `${API_BASE}/chat/stream`);
        xhr.setRequestHeader("Content-Type", "application/json");

        let lastIndex = 0;     // how far we've consumed in responseText
        let sseBuffer = "";    // carries an incomplete SSE event to the next onprogress

        const processSseText = (text: string) => {
            // Normalize SSE line endings; sse-starlette can send CRLF on the wire.
            const newData = (sseBuffer + text).replace(/\r\n/g, "\n").replace(/\r/g, "\n");

            // Split by the SSE event delimiter "\n\n"
            const events = newData.split("\n\n");

            // The last element may be an INCOMPLETE event — save it for next time
            sseBuffer = events.pop() ?? "";

            for (const event of events) {
                if (!event.trim()) continue;

                const lines = event.split("\n");
                let eventType = "message";
                let dataLine = "";

                for (const line of lines) {
                    if (line.startsWith("event: ") || line.startsWith("event:")) {
                        eventType = line.replace(/^event:\s*/, "").trim();
                    } else if (line.startsWith("data: ") || line.startsWith("data:")) {
                        dataLine = line.replace(/^data:\s*/, "").trim();
                    }
                }

                if (eventType === "done") {
                    return; // stream complete, onload will resolve
                }

                if (eventType === "error" && dataLine) {
                    try {
                        const parsed = JSON.parse(dataLine);
                        reject(new Error(parsed.detail ?? "Unknown stream error"));
                    } catch {
                        reject(new Error(dataLine));
                    }
                    return;
                }

                if (eventType === "message" && dataLine) {
                    try {
                        const parsed = JSON.parse(dataLine);
                        if (parsed.chunk) {
                            onChunk(parsed.chunk);
                        }
                    } catch {
                        // malformed JSON — skip
                    }
                }
            }
        };

        xhr.onprogress = () => {
            const text = xhr.responseText.substring(lastIndex);
            lastIndex = xhr.responseText.length;
            processSseText(text);
        };

        xhr.onload = () => {
            if (lastIndex < xhr.responseText.length) {
                processSseText(xhr.responseText.substring(lastIndex));
            } else if (sseBuffer.trim()) {
                processSseText("\n\n");
            }

            if (xhr.status >= 200 && xhr.status < 300) {
                resolve();
            } else {
                reject(new Error(`Backend error: ${xhr.status} ${xhr.statusText}`));
            }
        };

        xhr.onerror = () => {
            reject(new Error("Network error — could not reach backend"));
        };

        xhr.ontimeout = () => {
            reject(new Error("Request timed out — Space may be cold starting"));
        };

        // 2 minute timeout for cold starts
        xhr.timeout = 120000;

        xhr.send(JSON.stringify({ messages }));
    });
};

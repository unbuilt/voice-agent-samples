import {
  VoiceLiveClient,
  type VoiceLiveSession,
  type VoiceLiveSubscription,
} from "@azure/ai-voicelive";
import type { AccessToken, TokenCredential } from "@azure/core-auth";

/** Simple TokenCredential that wraps a pre-obtained access token. */
class StaticTokenCredential implements TokenCredential {
  constructor(private token: string) {}
  async getToken(): Promise<AccessToken> {
    return { token: this.token, expiresOnTimestamp: Date.now() + 3600_000 };
  }
}

// ---------------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------------
const $messages = document.getElementById("messages")!;
const $events = document.getElementById("events")!;
const $badge = document.getElementById("connectionBadge")!;
const $btnConnect = document.getElementById("btnConnect") as HTMLButtonElement;
const $btnDisconnect = document.getElementById("btnDisconnect") as HTMLButtonElement;
const $hint = document.getElementById("listeningHint")!;
const $voiceDot = document.getElementById("voiceStatusDot")!;
const $voiceText = document.getElementById("voiceStatusText")!;
const $transferLog = document.getElementById("transferLog")!;
const $agentPipeline = document.getElementById("agentPipeline")!;

const $cfgEndpoint = document.getElementById("cfgEndpoint") as HTMLInputElement;
const $cfgAgentName = document.getElementById("cfgAgentName") as HTMLInputElement;
const $cfgProjectName = document.getElementById("cfgProjectName") as HTMLInputElement;
const $cfgToken = document.getElementById("cfgToken") as HTMLInputElement;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let session: VoiceLiveSession | null = null;
let subscription: VoiceLiveSubscription | null = null;
let audioContext: AudioContext | null = null;
let audioQueue: AudioBuffer[] = [];
let isPlayingAudio = false;
let nextAudioStartTime = 0;
let currentSources: AudioBufferSourceNode[] = [];

let currentAssistantText = "";
let currentAssistantEl: HTMLElement | null = null;
let currentAgent = "triage_agent";
let transferCount = 0;

// Persist config across reloads
const STORAGE_KEY = "handoff-voicelive-config";
function loadConfig() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) {
      const c = JSON.parse(saved);
      if (c.endpoint) $cfgEndpoint.value = c.endpoint;
      if (c.agentName) $cfgAgentName.value = c.agentName;
      if (c.projectName) $cfgProjectName.value = c.projectName;
      if (c.token) $cfgToken.value = c.token;
    }
  } catch { /* ignore */ }
}
function saveConfig() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify({
    endpoint: $cfgEndpoint.value,
    agentName: $cfgAgentName.value,
    projectName: $cfgProjectName.value,
    token: $cfgToken.value,
  }));
}
loadConfig();

// ---------------------------------------------------------------------------
// UI helpers
// ---------------------------------------------------------------------------
function setConnectionState(state: "disconnected" | "connecting" | "connected") {
  $badge.className = `badge-${state}`;
  $badge.textContent = state.charAt(0).toUpperCase() + state.slice(1);
  $btnConnect.disabled = state !== "disconnected";
  $btnDisconnect.disabled = state === "disconnected";

  // Lock/unlock config inputs
  const locked = state !== "disconnected";
  $cfgEndpoint.disabled = locked;
  $cfgAgentName.disabled = locked;
  $cfgProjectName.disabled = locked;
  $cfgToken.disabled = locked;
}

function setVoiceStatus(status: "idle" | "listening" | "processing" | "speaking") {
  $voiceDot.className = status === "idle" ? "" : status;
  const labels: Record<string, string> = {
    idle: "Idle",
    listening: "Listening...",
    processing: "Processing...",
    speaking: "Agent speaking...",
  };
  $voiceText.textContent = labels[status] ?? status;
  $hint.textContent = status === "listening" ? "🎤 Speak now" : "";
}

function addMessage(role: "user" | "assistant" | "status", text: string): HTMLElement {
  const wrapper = document.createElement("div");
  if (role === "status") {
    wrapper.className = "msg msg-status";
    wrapper.textContent = text;
  } else {
    wrapper.className = `msg-wrapper-${role}`;
    const label = document.createElement("div");
    label.className = "msg-label";
    label.textContent = role === "user" ? "You" : agentDisplayName(currentAgent);
    const bubble = document.createElement("div");
    bubble.className = `msg msg-${role}`;
    bubble.textContent = text;
    wrapper.append(label, bubble);
  }
  $messages.appendChild(wrapper);
  $messages.scrollTop = $messages.scrollHeight;
  return wrapper;
}

function agentDisplayName(key: string): string {
  return key.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}

function logEvent(type: string, detail?: string) {
  const entry = document.createElement("div");
  entry.className = "event-entry";
  const ts = new Date().toLocaleTimeString("en-US", { hour12: false });
  entry.innerHTML = `<span class="event-type">${ts}</span> ${type}${detail ? " — " + detail : ""}`;
  $events.appendChild(entry);
  $events.scrollTop = $events.scrollHeight;
}

function setActiveAgent(agentKey: string) {
  currentAgent = agentKey;
  $agentPipeline.querySelectorAll(".agent-node").forEach(node => {
    node.classList.toggle("active", (node as HTMLElement).dataset.agent === agentKey);
  });
}

function addTransfer(from: string, to: string) {
  transferCount++;
  if (transferCount === 1) $transferLog.innerHTML = "";
  const entry = document.createElement("div");
  entry.className = "transfer-entry";
  const ts = new Date().toLocaleTimeString("en-US", { hour12: false });
  entry.innerHTML = `
    <div class="transfer-time">${ts}</div>
    <div class="transfer-detail">${agentDisplayName(from)} → ${agentDisplayName(to)}</div>
  `;
  $transferLog.prepend(entry);
}

// ---------------------------------------------------------------------------
// Handoff detection — parse response text for transfer signals
// ---------------------------------------------------------------------------
const AGENT_KEYS = ["triage_agent", "refund_agent", "order_agent"];

// Voice mapping per agent — mirrors the Python AGENT_VOICE_MAP
const AGENT_VOICE_MAP: Record<string, string> = {
  triage_agent: "en-US-Ava:DragonHDLatestNeural",
  refund_agent: "en-US-Brian:DragonHDLatestNeural",
  order_agent: "en-US-Emma:DragonHDLatestNeural",
};

function detectHandoff(text: string): string | null {
  const lower = text.toLowerCase();
  for (const key of AGENT_KEYS) {
    const label = key.replace(/_/g, " ");
    if (
      lower.includes(`transferring to ${label}`) ||
      lower.includes(`transferred to ${label}`) ||
      lower.includes(`transferring to ${key}`) ||
      lower.includes(`transferred to ${key}`)
    ) {
      return key;
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// Audio capture (microphone → session)
// ---------------------------------------------------------------------------
let captureStream: MediaStream | null = null;
let captureWorkletNode: AudioWorkletNode | null = null;

async function startCapture() {
  if (!session || !audioContext) return;
  captureStream = await navigator.mediaDevices.getUserMedia({
    audio: { sampleRate: 24000, channelCount: 1, echoCancellation: true, noiseSuppression: true },
  });

  // Use AudioWorklet for efficient low-latency capture
  const workletCode = `
    class CaptureProcessor extends AudioWorkletProcessor {
      process(inputs) {
        const input = inputs[0];
        if (input && input[0]) {
          // Convert Float32 [-1,1] to Int16 PCM
          const float32 = input[0];
          const int16 = new Int16Array(float32.length);
          for (let i = 0; i < float32.length; i++) {
            const s = Math.max(-1, Math.min(1, float32[i]));
            int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
          }
          this.port.postMessage(int16.buffer, [int16.buffer]);
        }
        return true;
      }
    }
    registerProcessor('capture-processor', CaptureProcessor);
  `;
  const blob = new Blob([workletCode], { type: "application/javascript" });
  const url = URL.createObjectURL(blob);
  await audioContext.audioWorklet.addModule(url);
  URL.revokeObjectURL(url);

  const source = audioContext.createMediaStreamSource(captureStream);
  captureWorkletNode = new AudioWorkletNode(audioContext, "capture-processor");
  captureWorkletNode.port.onmessage = (e: MessageEvent) => {
    if (session?.isConnected) {
      session.sendAudio(new Uint8Array(e.data)).catch(() => {});
    }
  };
  source.connect(captureWorkletNode);
  // Don't connect to destination — we don't want to echo
}

function stopCapture() {
  captureWorkletNode?.disconnect();
  captureWorkletNode = null;
  captureStream?.getTracks().forEach(t => t.stop());
  captureStream = null;
}

// ---------------------------------------------------------------------------
// Audio playback (session → speakers)
// ---------------------------------------------------------------------------
function clearAudioQueue() {
  audioQueue = [];
  isPlayingAudio = false;
  nextAudioStartTime = 0;
  for (const src of currentSources) {
    try { src.stop(); } catch { /* ignore */ }
  }
  currentSources = [];
}

function queueAudioDelta(delta: unknown) {
  if (!audioContext) return;

  let int16: Int16Array;

  if (delta instanceof ArrayBuffer) {
    int16 = new Int16Array(delta);
  } else if (delta instanceof Uint8Array) {
    // Ensure aligned copy for Int16Array view
    const aligned = new ArrayBuffer(delta.byteLength);
    new Uint8Array(aligned).set(delta);
    int16 = new Int16Array(aligned);
  } else if (typeof delta === "string") {
    // base64-encoded PCM16
    const raw = atob(delta);
    const bytes = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
    int16 = new Int16Array(bytes.buffer);
  } else {
    console.warn("[audio] Unknown delta type:", typeof delta, delta);
    return;
  }

  if (int16.length === 0) return;

  const float32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) {
    float32[i] = int16[i] / 0x8000;
  }
  const buf = audioContext.createBuffer(1, float32.length, 24000);
  buf.copyToChannel(float32, 0);

  const source = audioContext.createBufferSource();
  source.buffer = buf;
  source.connect(audioContext.destination);

  const now = audioContext.currentTime;
  const startAt = Math.max(now + 0.01, nextAudioStartTime);
  source.start(startAt);
  nextAudioStartTime = startAt + buf.duration;

  currentSources.push(source);
  source.onended = () => {
    const idx = currentSources.indexOf(source);
    if (idx >= 0) currentSources.splice(idx, 1);
  };
}

// ---------------------------------------------------------------------------
// Connect / Disconnect
// ---------------------------------------------------------------------------
async function connect() {
  const endpoint = $cfgEndpoint.value.trim();
  const agentName = $cfgAgentName.value.trim();
  const projectName = $cfgProjectName.value.trim();
  const token = $cfgToken.value.trim();

  if (!endpoint || !agentName || !projectName || !token) {
    alert("Please fill in all connection fields.");
    return;
  }

  saveConfig();
  setConnectionState("connecting");
  logEvent("connect", "Initiating connection...");

  try {
    audioContext = new AudioContext({ sampleRate: 24000 });
    // Browsers suspend AudioContext until user gesture — resume explicitly
    await audioContext.resume();

    const credential = new StaticTokenCredential(token);
    const client = new VoiceLiveClient(endpoint, credential);

    session = client.createSession({
      agent: {
        agentName,
        projectName,
      },
    });

    // Subscribe BEFORE connect so SESSION_UPDATED is not missed
    subscription = session.subscribe({
      onSessionUpdated: async (event: any) => {
        const s = event.session;
        const agent = s?.agent;
        logEvent("session.updated", `id=${s?.id ?? "?"} agent=${agent?.name ?? "?"}`);
      },

      onConversationItemInputAudioTranscriptionCompleted: async (event: any) => {
        const transcript = event.transcript ?? "";
        if (transcript.trim()) {
          addMessage("user", transcript);
          logEvent("user.transcript", transcript.substring(0, 60));
        }
      },

      onResponseTextDone: async (event: any) => {
        const text = event.text ?? "";
        if (text.trim()) {
          // Check for handoff in text responses
          checkAndHandleTransfer(text);
          addMessage("status", text);
          logEvent("response.text.done", text.substring(0, 60));
        }
      },

      onResponseAudioTranscriptDelta: async (event: any) => {
        const delta = event.delta ?? "";
        if (delta) {
          if (!currentAssistantEl) {
            currentAssistantText = "";
            currentAssistantEl = addMessage("assistant", "");
          }
          currentAssistantText += delta;
          const bubble = currentAssistantEl.querySelector(".msg-assistant");
          if (bubble) bubble.textContent = currentAssistantText;
          $messages.scrollTop = $messages.scrollHeight;
        }
      },

      onResponseAudioTranscriptDone: async (event: any) => {
        const transcript = event.transcript ?? "";
        logEvent("response.audio.transcript.done", transcript.substring(0, 60));

        // Check for handoff in audio transcripts
        checkAndHandleTransfer(transcript);

        // Finalize current assistant message
        currentAssistantEl = null;
        currentAssistantText = "";
      },

      onInputAudioBufferSpeechStarted: async () => {
        setVoiceStatus("listening");
        logEvent("speech.started");
        clearAudioQueue();
      },

      onInputAudioBufferSpeechStopped: async () => {
        setVoiceStatus("processing");
        logEvent("speech.stopped");
      },

      onResponseCreated: async () => {
        setVoiceStatus("speaking");
        logEvent("response.created");
      },

      onResponseAudioDelta: async (event: any) => {
        if (event.delta) {
          queueAudioDelta(event.delta);
          setVoiceStatus("speaking");
        }
      },

      onResponseAudioDone: async () => {
        setVoiceStatus("listening");
        logEvent("response.audio.done");
      },

      onResponseDone: async () => {
        setVoiceStatus("listening");
        logEvent("response.done");
        currentAssistantEl = null;
        currentAssistantText = "";
      },

      onServerError: async (event: any) => {
        const msg = event.error?.message ?? "Unknown error";
        if (msg.includes("no active response")) return;
        logEvent("error", msg);
      },

      onConversationItemCreated: async (event: any) => {
        logEvent("item.created", event.item?.id ?? "");
      },
    });

    await session.connect();
    logEvent("connected", "WebSocket open");

    // Configure session for audio
    await session.updateSession({
      modalities: ["text", "audio"],
      inputAudioFormat: "pcm16",
      outputAudioFormat: "pcm16",
      turnDetection: {
        type: "azure_semantic_vad",
        threshold: 0.5,
        prefixPaddingInMs: 300,
        silenceDurationInMs: 500,
      },
      inputAudioEchoCancellation: { type: "server_echo_cancellation" },
      inputAudioNoiseReduction: { type: "azure_deep_noise_suppression" },
    });
    logEvent("session.configured");

    // Send proactive greeting
    await session.sendEvent({ type: "response.create" });

    // Start mic capture
    await startCapture();

    setConnectionState("connected");
    setVoiceStatus("listening");
    setActiveAgent("triage_agent");
    logEvent("ready", "Mic active, waiting for speech");
  } catch (err: any) {
    setConnectionState("disconnected");
    setVoiceStatus("idle");
    logEvent("error", err.message ?? String(err));
    alert(`Connection failed: ${err.message ?? err}`);
  }
}

function checkAndHandleTransfer(text: string) {
  const target = detectHandoff(text);
  if (target && target !== currentAgent) {
    const from = currentAgent;
    addMessage("status", `🔄 ${agentDisplayName(from)} → ${agentDisplayName(target)}`);
    addTransfer(from, target);
    setActiveAgent(target);
    logEvent("handoff", `${from} → ${target}`);

    // Switch TTS voice for the new agent
    const voice = AGENT_VOICE_MAP[target];
    if (voice && session?.isConnected) {
      session.updateSession({ voice: { type: "azure-standard", name: voice } }).then(() => {
        logEvent("voice.switched", `${voice}`);
      }).catch((err: any) => {
        logEvent("voice.switch.error", err.message ?? String(err));
      });
    }
  }
}

async function disconnect() {
  logEvent("disconnect", "Cleaning up...");
  stopCapture();
  clearAudioQueue();

  if (subscription) {
    try { await subscription.close(); } catch { /* ignore */ }
    subscription = null;
  }
  if (session) {
    try { await session.disconnect(); } catch { /* ignore */ }
    try { await session.dispose(); } catch { /* ignore */ }
    session = null;
  }
  if (audioContext) {
    try { await audioContext.close(); } catch { /* ignore */ }
    audioContext = null;
  }

  setConnectionState("disconnected");
  setVoiceStatus("idle");
  setActiveAgent("triage_agent");
  currentAssistantEl = null;
  currentAssistantText = "";
  logEvent("disconnected");
}

// ---------------------------------------------------------------------------
// Button handlers
// ---------------------------------------------------------------------------
$btnConnect.addEventListener("click", connect);
$btnDisconnect.addEventListener("click", disconnect);

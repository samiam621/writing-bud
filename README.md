# Writing Buddy

**Writing Buddy is a Chrome extension that helps you write — in your own voice.**

Most AI writing tools make everything sound like... AI. Writing Buddy is different: you teach it *your* style first, by giving it samples of your own writing. From then on, whenever you ask it to write or rewrite something, it studies your voice — your word choice, rhythm, tone, quirks — and produces text that actually sounds like you wrote it.

## What Buddy does, in plain terms

1. You give Buddy a few pieces of your own writing (emails, essays, notes — anything).
2. Buddy reads them and remembers *how* you write.
3. Later, when you ask it to write something new or rewrite something on a webpage, it pulls up the parts of your past writing that are most similar to what you're asking for, and uses them as a style reference.
4. Gemini (Google's AI) then writes new content in that style — it imitates your *voice*, but never copies your old sentences or facts word-for-word.

## Getting started

### 1. Add your Gemini API key
Buddy runs on your own Gemini API key, so your requests use your own quota and stay private to you.
- Open the Buddy popup (click the extension icon).
- Paste your Gemini API key into the **"Your Gemini key"** section and hit **Save key**.
- Buddy checks the key with Gemini right away, so you'll know immediately if it's valid.
- Your key is stored only in your browser — never sent anywhere except to authenticate your own requests.

### 2. Teach Buddy your style
- In the **"Teach it your style"** section, upload a writing sample (`.txt`, `.md`, `.pdf`, or `.docx`) — something you actually wrote.
- Click **Upload sample**. Buddy reads it and adds it to your personal style memory.
- The more samples you add, the better Buddy understands your voice. You can upload as many as you like, whenever you like.

### 3. Ask Buddy to write
There are two ways to use Buddy:

**From the popup:**
- Type what you want written into the **"Ask it to write"** box (e.g. *"Write a short thank-you note to a mentor"*).
- Click **Generate**. Buddy writes a draft in your voice.
- Click **Copy to clipboard** and paste it wherever you need it.

**Directly on any webpage:**
- Highlight text in any editable field (an email, a document, a comment box).
- A small pencil icon will appear — click it.
- Choose **Rewrite this** (to restyle the selected text) or **Continue writing** (to keep going from where it leaves off).
- Optionally add a quick instruction (e.g. *"make it more formal"*).
- Review the draft in the preview, then click **Insert** to drop it right into the page.

### 4. Manage your data
- **Change key** — swap in a different Gemini key.
- **Remove key** — forget your key on this browser (your stored writing stays put on the server).
- **Delete my writing samples** — permanently erase everything you've taught Buddy. This is a two-click confirmation since it can't be undone.

## Good to know

- Buddy only ever imitates your *style* — it's instructed never to copy your original sentences or personal facts verbatim into a new draft.
- Nothing gets inserted into a page automatically. You always review the draft first and choose to Insert it yourself.
- Your writing samples and key are scoped to you — Buddy keeps every user's style memory separate.

## Coming soon

- Writing directly inside Google Docs, not just copy/paste
- Sign-in support so your style follows you across devices


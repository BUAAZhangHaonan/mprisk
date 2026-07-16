import { mediaUrl, Sample } from "../api";

type View = "M1" | "M2" | "M12";
type MediaGroup = "visual" | "audio" | "text" | "unknown";
type MediaKind = "image" | "video" | "audio" | "text" | "path";

type MediaEntry = {
  key: string;
  value: string;
  group: MediaGroup;
  kind: MediaKind;
  viewHint: View | null;
};

const imageExtensions = new Set(["avif", "bmp", "gif", "jpeg", "jpg", "png", "svg", "webp"]);
const videoExtensions = new Set(["avi", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ogg", "webm"]);
const audioExtensions = new Set(["aac", "flac", "m4a", "mp3", "oga", "opus", "wav", "weba"]);

const protocolModalities: Record<string, { M1: MediaGroup; M2: MediaGroup }> = {
  VT: { M1: "visual", M2: "text" },
  VA: { M1: "visual", M2: "audio" },
  IT: { M1: "visual", M2: "text" },
};

function extensionOf(value: string) {
  const withoutQuery = value.split(/[?#]/, 1)[0] ?? "";
  const filename = withoutQuery.split(/[\\/]/).pop() ?? withoutQuery;
  const match = /\.([a-z0-9]+)$/i.exec(filename);
  return match?.[1]?.toLowerCase() ?? "";
}

function normalizeGroup(value?: string): MediaGroup | null {
  const normalized = value?.toLowerCase() ?? "";
  if (/(vision|visual|image|video)/.test(normalized)) return "visual";
  if (/audio|speech|sound/.test(normalized)) return "audio";
  if (/text|transcript|caption|language/.test(normalized)) return "text";
  return null;
}

function inferViewHint(key: string): View | null {
  const normalized = key.toLowerCase();
  if (/(^|[_-])(m12|joint|multi|combined|fusion)([_-]|$)/.test(normalized)) return "M12";
  if (/(^|[_-])(m1|modality1|modality_1|first)([_-]|$)/.test(normalized)) return "M1";
  if (/(^|[_-])(m2|modality2|modality_2|second)([_-]|$)/.test(normalized)) return "M2";
  return null;
}

function inferKind(key: string, value: string): MediaKind {
  const extension = extensionOf(value);
  if (imageExtensions.has(extension)) return "image";
  if (videoExtensions.has(extension)) return "video";
  if (audioExtensions.has(extension)) return "audio";

  const group = normalizeGroup(key);
  if (group === "audio") return "audio";
  if (group === "text") return "text";
  return "path";
}

function inferGroup(key: string, kind: MediaKind): MediaGroup {
  const group = normalizeGroup(key);
  if (group) return group;
  if (kind === "image" || kind === "video") return "visual";
  if (kind === "audio") return "audio";
  if (kind === "text") return "text";
  return "unknown";
}

function looksLikePath(value: string) {
  const trimmed = value.trim();
  return (
    /^(?:\.{1,2}[\\/]|[\\/]|[a-zA-Z]:[\\/]|https?:\/\/)/.test(trimmed) ||
    /[\\/]/.test(trimmed) ||
    /^[^\s]+\.(?:csv|json|jsonl|md|srt|txt|vtt)$/i.test(trimmed)
  );
}

function isInlineText(value: string) {
  return value.trim().length > 0 && !looksLikePath(value);
}

function mediaEntries(sample: Sample): MediaEntry[] {
  return Object.entries(sample.media_paths ?? {})
    .filter(([, value]) => value.trim().length > 0)
    .map(([key, value]) => {
      const kind = inferKind(key, value);
      return {
        key,
        value,
        kind,
        group: inferGroup(key, kind),
        viewHint: inferViewHint(key),
      };
    });
}

function protocolGroups(sample: Sample) {
  const protocol = protocolModalities[sample.protocol?.toUpperCase()] ?? null;
  return {
    M1: normalizeGroup(sample.m1_modality) ?? protocol?.M1 ?? null,
    M2: normalizeGroup(sample.m2_modality) ?? protocol?.M2 ?? null,
  };
}

function mediaForView(sample: Sample, view: View) {
  const entries = mediaEntries(sample);
  const groups = protocolGroups(sample);
  const hasJointMedia = entries.some((entry) => entry.viewHint === "M12");

  return entries.filter((entry) => {
    if (entry.viewHint) return entry.viewHint === view;
    if (view === "M12") {
      if (hasJointMedia) return false;
      if (!groups.M1 && !groups.M2) return true;
      return entry.group === groups.M1 || entry.group === groups.M2;
    }

    const targetGroup = groups[view];
    if (targetGroup) return entry.group === targetGroup;
    if (view === "M1") return entry.group === "visual";
    return entry.group === "text" || entry.group === "audio";
  });
}

function renderMedia(entry: MediaEntry) {
  if (entry.kind === "image") {
    return <img className="mediaAsset" src={mediaUrl(entry.value)} alt={entry.key} />;
  }
  if (entry.kind === "video") {
    return <video className="mediaAsset" src={mediaUrl(entry.value)} controls preload="metadata" />;
  }
  if (entry.kind === "audio") {
    return <audio className="audioAsset" src={mediaUrl(entry.value)} controls preload="metadata" />;
  }
  if (entry.kind === "text" && isInlineText(entry.value)) {
    return <p className="inlineText">{entry.value}</p>;
  }
  return <code className="pathHint">{entry.value}</code>;
}

export function MediaPanel({ sample, view }: { sample: Sample; view: "M1" | "M2" | "M12" }) {
  const media = mediaForView(sample, view);

  return (
    <section className="mediaPanel">
      <div className="mediaTitle">{sample.sample_id} · {view}</div>
      {media.length > 0 ? (
        <div className="mediaGrid">
          {media.map((entry) => (
            <figure className="mediaBox" key={entry.key}>
              <figcaption>{entry.key}</figcaption>
              {renderMedia(entry)}
            </figure>
          ))}
        </div>
      ) : (
        <div className="mediaBox emptyMedia">No media for this view.</div>
      )}
    </section>
  );
}

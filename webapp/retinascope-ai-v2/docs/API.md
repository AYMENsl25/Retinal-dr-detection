# API

## Health

```http
GET /api/v1/health
```

Returns backend version, LLM provider, and model loading status.

## Analyze

```http
POST /api/v1/analyze
Content-Type: multipart/form-data
file=<fundus image>
```

Returns:

```text
case_id
5 panels
damage zoom crops
grade and grade probabilities
confidence and uncertainty
biomarkers
clinical report
vascular report
```

## Chat

```http
POST /api/v1/chat
```

Body:

```json
{
  "message": "What does the tortuosity score mean?",
  "case_context": {}
}
```

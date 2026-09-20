package com.demystifier.backend.service;

import com.demystifier.backend.dto.TextResultDto;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.client.WebClientResponseException;

import java.util.List;
import java.util.Map;

@Service
public class TextEngineClient {

    private static final String SYSTEM_PROMPT = """
            You are a medical report translator. Your ONLY job is to rephrase clinical
            language into plain, empathetic language a patient can understand, for a
            REPORT the patient has already received.

            Strict rules:
            - Never suggest a diagnosis, treatment, medication, or dosage.
            - Never contradict or add information not present in the source report.
            - Always end with: "Please discuss this report with your doctor for full context."
            - If the report is ambiguous, unreadable, or not a medical report, say so plainly.
            - Do not speculate about prognosis or severity beyond what the report states.

            Return ONLY plain text: a one-paragraph plain-language summary.
            """;

    private final WebClient llmWebClient;
    private final String apiKey;

    public TextEngineClient(WebClient llmWebClient, @Value("${llm.api.key}") String apiKey) {
        this.llmWebClient = llmWebClient;
        this.apiKey = apiKey;
    }

    public TextResultDto summarize(String reportText) {
        Map<String, Object> payload = Map.of(
                "systemInstruction", Map.of("parts", List.of(Map.of("text", SYSTEM_PROMPT))),
                "contents", List.of(Map.of("parts", List.of(Map.of("text", "Patient report:\n\n" + reportText)))),
                "generationConfig", Map.of("temperature", 0.1, "maxOutputTokens", 1024));

        Map response;
        try {
            response = llmWebClient.post()
                    .uri(uriBuilder -> uriBuilder.path("/gemini-3.6-flash:generateContent")
                            .queryParam("key", apiKey).build())
                    .bodyValue(payload)
                    .retrieve()
                    .bodyToMono(Map.class)
                    .block();
        } catch (WebClientResponseException e) {
             return new TextResultDto("Error calling LLM: " + e.getResponseBodyAsString(), List.of(), "");
        }

        String summary = extractText(response);
        return new TextResultDto(summary, List.of(), "Please discuss this report with your doctor for full context.");
    }

    @SuppressWarnings("unchecked")
    private String extractText(Map response) {
        try {
            var candidates = (List<Map>) response.get("candidates");
            var content = (Map) candidates.get(0).get("content");
            var parts = (List<Map>) content.get("parts");
            return (String) parts.get(0).get("text");
        } catch (Exception e) {
            return "Could not generate a summary — please try again.";
        }
    }
}
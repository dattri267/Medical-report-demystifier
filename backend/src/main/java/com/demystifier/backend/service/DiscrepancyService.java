package com.demystifier.backend.service;

import com.demystifier.backend.dto.DiscrepancyDto;
import com.demystifier.backend.dto.TextResultDto;
import com.demystifier.backend.dto.VisionResultDto;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Map;

@Service
public class DiscrepancyService {

    private static final double CONFIDENCE_THRESHOLD = 0.7;

    // If vision is confident about a disease, but the text contains one of these
    // phrases, that's a contradiction worth flagging.
    private static final Map<String, List<String>> CONTRADICTION_PHRASES = Map.of(
            "Pneumonia",
            List.of("clear lung", "no evidence of pneumonia", "lungs appear clear", "no acute", "unremarkable"),
            "Cardiomegaly", List.of("normal heart size", "normal cardiac silhouette", "heart size within normal"),
            "Pneumothorax", List.of("no pneumothorax", "lungs fully expanded", "no evidence of collapse"),
            "Effusion", List.of("no effusion", "no pleural effusion", "clear pleural"),
            "Atelectasis", List.of("no atelectasis", "lungs fully expanded"),
            "Mass", List.of("no mass", "no suspicious mass"),
            "Nodule", List.of("no nodule", "no pulmonary nodule"));

    public DiscrepancyDto check(VisionResultDto visionResult, TextResultDto textResult) {
        if (visionResult == null || textResult == null || textResult.getPlainLanguageSummary() == null) {
            return new DiscrepancyDto(false, "Not enough data to compare.");
        }

        String summaryLower = textResult.getPlainLanguageSummary().toLowerCase();

        for (Map.Entry<String, Double> entry : visionResult.getFindings().entrySet()) {
            String disease = entry.getKey();
            double confidence = entry.getValue();

            if (confidence >= CONFIDENCE_THRESHOLD && CONTRADICTION_PHRASES.containsKey(disease)) {
                for (String phrase : CONTRADICTION_PHRASES.get(disease)) {
                    if (summaryLower.contains(phrase)) {
                        String reason = String.format(
                                "Vision model flagged %s with %.0f%% confidence, but the report text suggests '%s'.",
                                disease, confidence * 100, phrase);
                        return new DiscrepancyDto(true, reason);
                    }
                }
            }
        }

        return new DiscrepancyDto(false, "No contradictions detected.");
    }
}
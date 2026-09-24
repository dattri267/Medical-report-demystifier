package com.demystifier.backend.service;

import com.demystifier.backend.config.LabelConstants;
import com.demystifier.backend.dto.VisionResultDto;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

@Service
@Profile("!real-vision") // active unless the "real-vision" profile is explicitly enabled
public class MockVisionClient implements VisionClient {

    private final Random random = new Random();

    @Override
    public VisionResultDto analyzeImage(MultipartFile image) {
        Map<String, Double> findings = new HashMap<>();
        for (String label : LabelConstants.LABELS) {
            findings.put(label, Math.round(random.nextDouble() * 0.3 * 100.0) / 100.0);
        }

        // force one finding to be prominent so discrepancy testing has something to
        // work with
        String topLabel = LabelConstants.LABELS.get(random.nextInt(LabelConstants.LABELS.size()));
        findings.put(topLabel, 0.85);

        VisionResultDto result = new VisionResultDto();
        result.setFindings(findings);
        result.setTopFinding(topLabel);
        result.setTopFindingConfidence(0.85);
        result.setGradcamOverlayB64(""); // no real image in mock mode

        return result;
    }
}
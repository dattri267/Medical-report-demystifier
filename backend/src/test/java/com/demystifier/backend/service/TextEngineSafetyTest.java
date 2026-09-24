package com.demystifier.backend.service;

import com.demystifier.backend.dto.TextResultDto;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import java.io.InputStream;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest
class TextEngineSafetyTest {

    @Autowired
    private TextEngineClient textEngineClient;

    record RedTeamCase(String id, String category, String input, List<String> mustContain,
            List<String> mustNotContain) {
    }

    private List<RedTeamCase> loadCases() throws Exception {
        ObjectMapper mapper = new ObjectMapper();
        try (InputStream is = getClass().getClassLoader().getResourceAsStream("redteam/cases.json")) {
            return mapper.readValue(is, mapper.getTypeFactory().constructCollectionType(List.class, RedTeamCase.class));
        }
    }

    @TestFactory
    Stream<DynamicTest> redTeamSafetyChecks() throws Exception {
        List<RedTeamCase> cases = loadCases();

        return cases.stream().map(c -> DynamicTest.dynamicTest(
                c.id() + " [" + c.category() + "]",
                () -> {
                    Thread.sleep(13000); // stay under Gemini free-tier rate limit (5 req/min)

                    TextResultDto result = textEngineClient.summarize(c.input());
                    String fullOutput = (result.getPlainLanguageSummary() + " " + result.getDisclaimer()).toLowerCase();

                    for (String required : c.mustContain()) {
                        assertTrue(fullOutput.contains(required.toLowerCase()),
                                "[" + c.id() + "] Expected output to contain '" + required
                                        + "' but it did not.\nOutput: " + fullOutput);
                    }

                    for (String forbidden : c.mustNotContain()) {
                        assertFalse(fullOutput.contains(forbidden.toLowerCase()),
                                "[" + c.id() + "] Output contained forbidden phrase '" + forbidden + "'.\nOutput: "
                                        + fullOutput);
                    }
                }));
    }
}
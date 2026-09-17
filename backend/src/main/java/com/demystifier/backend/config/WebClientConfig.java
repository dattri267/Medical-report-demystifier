package com.demystifier.backend.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.reactive.function.client.WebClient;

@Configuration
public class WebClientConfig {

    @Bean
    public WebClient visionWebClient(@Value("${vision.service.url}") String visionUrl) {
        return WebClient.builder().baseUrl(visionUrl).build();
    }

    @Bean
    public WebClient llmWebClient() {
        return WebClient.builder()
                .baseUrl("https://generativelanguage.googleapis.com/v1beta/models")
                .build();
    }
}
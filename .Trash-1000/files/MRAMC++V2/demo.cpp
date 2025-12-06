/*
 * ============================================================================
 * DEMO APPLICATION - Enhanced Marketing Agent
 * ============================================================================
 * 
 * This file demonstrates how to use the Enhanced Marketing Agent for:
 * 1. Single message response prediction
 * 2. Multi-touchpoint campaign analysis
 * 3. Individual element testing
 * 4. Model training
 * 
 * ============================================================================
 */

#include "enhanced_marketing_agent.hpp"
#include <iostream>
#include <iomanip>

using namespace marketing;

// Helper function to print section headers
void print_section(const std::string& title) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(70, '=') << "\n\n";
}

// Demo 1: Basic message response prediction
void demo_basic_prediction() {
    print_section("DEMO 1: Basic Message Response Prediction");
    
    // Create the agent
    EnhancedMarketingAgent agent;
    
    // Create a marketing message
    Message message;
    message.content = "Limited time offer! Get 50% off our premium subscription. "
                     "Join thousands of happy customers today!";
    message.channel = "email";
    message.features["urgency"] = 0.8;
    message.features["discount"] = 0.5;
    
    // Create a target person
    Person person;
    person.age = 32;
    person.gender = "female";
    person.location = "New York";
    person.interests = {"technology", "shopping", "productivity"};
    person.behavioral_features["engagement_rate"] = 0.65;
    person.behavioral_features["previous_purchases"] = 3.0;
    
    // Predict response
    std::string response = agent.predict_response(message, person);
    
    std::cout << "Message: " << message.content.substr(0, 60) << "...\n";
    std::cout << "Channel: " << message.channel << "\n";
    std::cout << "Target: " << person.age << "yo " << person.gender 
              << " from " << person.location << "\n";
    std::cout << "\n🎯 Predicted Response: " << response << "\n";
}

// Demo 2: Campaign analysis with multiple touchpoints
void demo_campaign_analysis() {
    print_section("DEMO 2: Multi-Touchpoint Campaign Analysis");
    
    EnhancedMarketingAgent agent;
    
    // Create target person
    Person person;
    person.age = 28;
    person.gender = "male";
    person.location = "San Francisco";
    person.interests = {"fitness", "health", "technology"};
    person.behavioral_features["engagement_rate"] = 0.72;
    
    // Create campaign with multiple touchpoints
    std::vector<TouchPoint> campaign;
    
    // Touchpoint 1: Initial awareness (Day 0)
    Message msg1;
    msg1.content = "Discover the future of fitness tracking with our new smartwatch!";
    msg1.channel = "social";
    campaign.emplace_back(msg1, 0.0, "awareness");
    
    // Touchpoint 2: Consideration (Day 2)
    Message msg2;
    msg2.content = "See what fitness experts are saying about our smartwatch. "
                   "Track your workouts, monitor your health, achieve your goals!";
    msg2.channel = "email";
    campaign.emplace_back(msg2, 2.0, "consideration");
    
    // Touchpoint 3: Decision (Day 5)
    Message msg3;
    msg3.content = "Ready to transform your fitness journey? Get 20% off today only!";
    msg3.channel = "email";
    campaign.emplace_back(msg3, 5.0, "decision");
    
    // Touchpoint 4: Retargeting (Day 7)
    Message msg4;
    msg4.content = "Don't miss out! Your 20% discount expires in 24 hours.";
    msg4.channel = "display";
    campaign.emplace_back(msg4, 7.0, "decision");
    
    // Analyze campaign
    CampaignAnalysis analysis = agent.analyze_campaign(campaign, person);
    
    std::cout << "Campaign: 4 touchpoints over 7 days\n";
    std::cout << "Target: " << person.age << "yo " << person.gender << "\n\n";
    
    std::cout << "📊 Campaign Results:\n";
    std::cout << "  Overall Effectiveness: " << std::fixed << std::setprecision(2) 
              << (analysis.overall_effectiveness * 100) << "%\n";
    std::cout << "  Temporal Decay Factor: " << analysis.temporal_decay_factor << "\n\n";
    
    std::cout << "📈 Channel Performance:\n";
    for (const auto& [channel, performance] : analysis.channel_performance) {
        std::cout << "  " << channel << ": " << (performance * 100) << "%\n";
    }
    
    std::cout << "\n🛤️  Customer Journey:\n";
    for (const auto& [stage, progression] : analysis.journey_progression) {
        std::cout << "  " << stage << ": " << (progression * 100) << "%\n";
    }
    
    std::cout << "\n🔗 Channel Synergy Scores:\n";
    for (const auto& [channel, synergy] : analysis.synergy_scores) {
        std::cout << "  " << channel << ": " << synergy << "\n";
    }
    
    if (!analysis.recommendations.empty()) {
        std::cout << "\n💡 Recommendations:\n";
        for (size_t i = 0; i < analysis.recommendations.size(); ++i) {
            std::cout << "  " << (i + 1) << ". " << analysis.recommendations[i] << "\n";
        }
    }
}

// Demo 3: A/B testing message elements
void demo_element_testing() {
    print_section("DEMO 3: A/B Testing Message Elements");
    
    EnhancedMarketingAgent agent;
    
    std::cout << "Testing different headlines for the same product:\n\n";
    
    // Test multiple headline variations
    std::vector<MessageElement> headlines = {
        {"headline", "Save 50% Today Only!"},
        {"headline", "Transform Your Life with Our Premium Service"},
        {"headline", "Join 10,000+ Happy Customers"},
        {"headline", "Limited Time: Exclusive Offer Inside"},
        {"headline", "The Smart Choice for Modern Living"}
    };
    
    std::cout << std::left << std::setw(50) << "Headline" 
              << std::right << std::setw(15) << "Score" << "\n";
    std::cout << std::string(65, '-') << "\n";
    
    for (const auto& headline : headlines) {
        double score = agent.analyze_element(headline);
        std::cout << std::left << std::setw(50) << headline.content 
                  << std::right << std::setw(14) << std::fixed 
                  << std::setprecision(2) << (score * 100) << "%\n";
    }
    
    // Test different CTA buttons
    std::cout << "\n\nTesting different Call-To-Action buttons:\n\n";
    
    std::vector<MessageElement> ctas = {
        {"cta", "Buy Now"},
        {"cta", "Get Started Free"},
        {"cta", "Learn More"},
        {"cta", "Claim Your Discount"},
        {"cta", "Try It Risk-Free"}
    };
    
    std::cout << std::left << std::setw(50) << "CTA Button" 
              << std::right << std::setw(15) << "Score" << "\n";
    std::cout << std::string(65, '-') << "\n";
    
    for (const auto& cta : ctas) {
        double score = agent.analyze_element(cta);
        std::cout << std::left << std::setw(50) << cta.content 
                  << std::right << std::setw(14) << std::fixed 
                  << std::setprecision(2) << (score * 100) << "%\n";
    }
}

// Demo 4: Training the agent on historical data
void demo_training() {
    print_section("DEMO 4: Training on Historical Data");
    
    EnhancedMarketingAgent agent;
    
    // Create training dataset
    std::vector<Message> messages;
    std::vector<Person> people;
    std::vector<std::string> responses;
    
    // Training example 1: High engagement message
    Message msg1;
    msg1.content = "Exclusive offer just for you! Save 40% on your favorite products.";
    msg1.channel = "email";
    messages.push_back(msg1);
    
    Person person1;
    person1.age = 35;
    person1.gender = "female";
    person1.behavioral_features["engagement_rate"] = 0.75;
    people.push_back(person1);
    responses.push_back("convert");
    
    // Training example 2: Low engagement message
    Message msg2;
    msg2.content = "Newsletter: Here's what's new this week.";
    msg2.channel = "email";
    messages.push_back(msg2);
    
    Person person2;
    person2.age = 42;
    person2.gender = "male";
    person2.behavioral_features["engagement_rate"] = 0.32;
    people.push_back(person2);
    responses.push_back("ignore");
    
    // Training example 3: Moderate engagement
    Message msg3;
    msg3.content = "New features available! Check out what's new in our latest update.";
    msg3.channel = "social";
    messages.push_back(msg3);
    
    Person person3;
    person3.age = 28;
    person3.gender = "female";
    person3.behavioral_features["engagement_rate"] = 0.58;
    people.push_back(person3);
    responses.push_back("engage");
    
    std::cout << "Training dataset: " << messages.size() << " examples\n";
    std::cout << "Response distribution:\n";
    std::cout << "  - Convert: 1\n";
    std::cout << "  - Engage: 1\n";
    std::cout << "  - Ignore: 1\n\n";
    
    // Train the agent
    std::cout << "Starting training process...\n\n";
    agent.train(messages, people, responses, 50, 0.001);
    
    std::cout << "\n✅ Training completed!\n";
    std::cout << "Models can be saved using agent.save_models(\"./models\");\n";
}

// Demo 5: Real-world scenario - E-commerce campaign
void demo_ecommerce_scenario() {
    print_section("DEMO 5: Real-World E-Commerce Scenario");
    
    EnhancedMarketingAgent agent;
    
    std::cout << "Scenario: Online clothing retailer running a seasonal sale campaign\n\n";
    
    // Customer profile
    Person customer;
    customer.age = 29;
    customer.gender = "female";
    customer.location = "Los Angeles";
    customer.interests = {"fashion", "shopping", "lifestyle"};
    customer.behavioral_features["previous_purchases"] = 5.0;
    customer.behavioral_features["avg_order_value"] = 120.0;
    customer.behavioral_features["days_since_last_purchase"] = 45.0;
    customer.behavioral_features["email_open_rate"] = 0.68;
    
    std::cout << "Customer Profile:\n";
    std::cout << "  Age: " << customer.age << "\n";
    std::cout << "  Location: " << customer.location << "\n";
    std::cout << "  Previous Purchases: " << customer.behavioral_features["previous_purchases"] << "\n";
    std::cout << "  Email Open Rate: " << (customer.behavioral_features["email_open_rate"] * 100) << "%\n\n";
    
    // Test different message strategies
    std::vector<Message> strategies;
    
    Message strategy1;
    strategy1.content = "🎉 Summer Sale: 30% off everything! Shop now before it's gone!";
    strategy1.channel = "email";
    strategy1.features["urgency"] = 0.9;
    strategy1.features["discount"] = 0.3;
    strategies.push_back(strategy1);
    
    Message strategy2;
    strategy2.content = "New arrivals perfect for your style. Curated just for you based on your favorites.";
    strategy2.channel = "email";
    strategy2.features["personalization"] = 0.85;
    strategies.push_back(strategy2);
    
    Message strategy3;
    strategy3.content = "Your favorite brands are back in stock! Plus, free shipping on orders over $50.";
    strategy3.channel = "email";
    strategy3.features["value_prop"] = 0.75;
    strategies.push_back(strategy3);
    
    std::cout << "Testing 3 different message strategies:\n\n";
    std::cout << std::left << std::setw(10) << "Strategy" 
              << std::setw(45) << "Message Preview" 
              << std::right << std::setw(15) << "Response" << "\n";
    std::cout << std::string(70, '-') << "\n";
    
    for (size_t i = 0; i < strategies.size(); ++i) {
        std::string response = agent.predict_response(strategies[i], customer);
        std::string preview = strategies[i].content.substr(0, 42) + "...";
        
        std::cout << std::left << std::setw(10) << ("#" + std::to_string(i + 1))
                  << std::setw(45) << preview
                  << std::right << std::setw(15) << response << "\n";
    }
    
    std::cout << "\n💡 Insight: Test all strategies and select the one with highest predicted conversion!\n";
}

// Main function - run all demos
int main() {
    std::cout << R"(
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║         ENHANCED MARKETING RESPONSE AGENT - DEMO APP              ║
    ║                                                                   ║
    ║         AI-Powered Marketing Response Prediction System           ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    )" << std::endl;
    
    try {
        // Run all demos
        demo_basic_prediction();
        demo_campaign_analysis();
        demo_element_testing();
        demo_training();
        demo_ecommerce_scenario();
        
        print_section("Demo Complete");
        std::cout << "✅ All demos executed successfully!\n\n";
        std::cout << "Next steps:\n";
        std::cout << "  1. Modify the demos to test your own marketing messages\n";
        std::cout << "  2. Train the agent on your historical data\n";
        std::cout << "  3. Integrate into your marketing automation system\n";
        std::cout << "  4. Monitor and optimize based on real results\n\n";
        
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
}

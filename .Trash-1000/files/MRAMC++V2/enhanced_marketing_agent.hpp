#ifndef ENHANCED_MARKETING_AGENT_HPP
#define ENHANCED_MARKETING_AGENT_HPP

#include <vector>
#include <string>
#include <map>
#include <memory>
#include <random>
#include <cmath>
#include <algorithm>
#include <stdexcept>

/*
 * ============================================================================
 * ENHANCED MARKETING RESPONSE AGENT - MODULE INDEX
 * ============================================================================
 * 
 * Quick Navigation Guide:
 * 
 * MODULE 1: CORE DATA STRUCTURES (Lines 50-180)
 *   Purpose: Fundamental data types for marketing messages and people
 *   Classes: Message, Person, MessageElement, CampaignAnalysis, TouchPoint
 * 
 * MODULE 2: NEURAL NETWORK FOUNDATION (Lines 180-320)
 *   Purpose: Basic neural network building blocks
 *   Classes: Neuron, Layer, NeuralNetwork
 * 
 * MODULE 3: SPECIALIZED NEURAL NETWORKS (Lines 320-550)
 *   Purpose: Domain-specific neural network architectures
 *   Classes:
 *     - CognitiveProcessingNetwork (message content analysis)
 *     - AffectiveProcessingNetwork (emotional impact)
 *     - DecisionNetwork (behavior prediction)
 *     - ElementAnalysisNetwork (component evaluation)
 *     - CampaignEffectivenessNetwork (LSTM for sequences)
 * 
 * MODULE 4: BAYESIAN NETWORK (Lines 550-650)
 *   Purpose: Probabilistic reasoning and uncertainty modeling
 *   Classes: BayesianNode, BayesianNetwork
 * 
 * MODULE 5: MAIN AGENT (Lines 650-end)
 *   Purpose: Integrated marketing response prediction system
 *   Classes: EnhancedMarketingAgent
 *   Key Methods:
 *     - predict_response()
 *     - analyze_campaign()
 *     - analyze_element()
 *     - train()
 * 
 * ============================================================================
 */

namespace marketing {

// ============================================================================
// MODULE 1: CORE DATA STRUCTURES
// ============================================================================
// This module defines the fundamental data types used throughout the system

/**
 * @brief Represents a marketing message with content and metadata
 */
struct Message {
    std::string content;
    std::string channel;  // email, social, display, etc.
    std::map<std::string, double> features;  // Extracted features
    
    Message(const std::string& c = "", const std::string& ch = "")
        : content(c), channel(ch) {}
};

/**
 * @brief Represents a person with demographic and behavioral attributes
 */
struct Person {
    int age;
    std::string gender;
    std::string location;
    std::vector<std::string> interests;
    std::map<std::string, double> behavioral_features;
    
    Person() : age(0), gender("unknown"), location("unknown") {}
};

/**
 * @brief Represents an individual element of a marketing message
 */
struct MessageElement {
    std::string type;  // headline, image, cta, body, etc.
    std::string content;
    std::map<std::string, double> features;
    
    MessageElement(const std::string& t = "", const std::string& c = "")
        : type(t), content(c) {}
};

/**
 * @brief Results of a campaign analysis
 */
struct CampaignAnalysis {
    double overall_effectiveness;
    std::map<std::string, double> channel_performance;
    std::map<std::string, double> journey_progression;
    std::map<std::string, double> synergy_scores;
    double temporal_decay_factor;
    std::vector<std::string> recommendations;
};

/**
 * @brief Represents a touchpoint in a customer journey
 */
struct TouchPoint {
    Message message;
    double time_offset;  // Time relative to campaign start
    std::string stage;   // awareness, consideration, decision, etc.
    
    TouchPoint() : time_offset(0.0) {}
    TouchPoint(const Message& m, double t, const std::string& s)
        : message(m), time_offset(t), stage(s) {}
};

// ============================================================================
// MODULE 2: NEURAL NETWORK FOUNDATION
// ============================================================================
// This module provides the basic building blocks for all neural networks

/**
 * @brief Single neuron with weights, bias, and activation function
 */
class Neuron {
private:
    std::vector<double> weights;
    double bias;
    std::string activation_type;
    
    double activate(double x) const;
    double activate_derivative(double x) const;

public:
    Neuron(int input_size, const std::string& activation = "relu");
    
    double forward(const std::vector<double>& inputs) const;
    void update_weights(const std::vector<double>& gradients, double learning_rate);
    
    const std::vector<double>& get_weights() const { return weights; }
    double get_bias() const { return bias; }
};

/**
 * @brief Layer of neurons with forward and backward propagation
 */
class Layer {
private:
    std::vector<Neuron> neurons;
    int input_size;
    std::string activation_type;
    
    // Cached values for backpropagation
    std::vector<double> last_input;
    std::vector<double> last_output;

public:
    Layer(int input_size, int output_size, const std::string& activation = "relu");
    
    std::vector<double> forward(const std::vector<double>& inputs);
    std::vector<double> backward(const std::vector<double>& grad_output, double learning_rate);
    
    int get_output_size() const { return static_cast<int>(neurons.size()); }
};

/**
 * @brief Multi-layer feedforward neural network
 */
class NeuralNetwork {
protected:
    std::vector<Layer> layers;
    std::mt19937 rng;

public:
    NeuralNetwork();
    virtual ~NeuralNetwork() = default;
    
    void add_layer(int input_size, int output_size, const std::string& activation = "relu");
    virtual std::vector<double> forward(const std::vector<double>& inputs);
    virtual void train(const std::vector<double>& inputs, 
                      const std::vector<double>& targets,
                      double learning_rate = 0.01);
    
    void save_model(const std::string& filepath) const;
    void load_model(const std::string& filepath);
};

// ============================================================================
// MODULE 3: SPECIALIZED NEURAL NETWORKS
// ============================================================================
// This module contains domain-specific neural network architectures for
// different aspects of marketing response prediction

/**
 * @brief Network for analyzing message content and extracting cognitive features
 * 
 * Architecture: Processes textual and visual content to understand
 * clarity, relevance, and information value
 */
class CognitiveProcessingNetwork : public NeuralNetwork {
private:
    int text_encoding_dim;

public:
    CognitiveProcessingNetwork(int encoding_dim = 128);
    
    std::vector<double> process(const Message& message);
    std::vector<double> encode_text(const std::string& text);
};

/**
 * @brief Network for analyzing emotional impact of messages
 * 
 * Architecture: Evaluates emotional tone, appeal, and psychological triggers
 */
class AffectiveProcessingNetwork : public NeuralNetwork {
private:
    int emotion_dim;

public:
    AffectiveProcessingNetwork(int emotion_dim = 64);
    
    std::vector<double> process(const Message& message);
    std::vector<double> extract_emotional_features(const std::string& content);
};

/**
 * @brief Network for predicting behavioral responses
 * 
 * Architecture: Combines cognitive and affective features with person
 * characteristics to predict response category
 */
class DecisionNetwork : public NeuralNetwork {
public:
    DecisionNetwork(int cognitive_dim, int affective_dim, int person_dim);
    
    std::vector<double> predict(const std::vector<double>& cognitive_features,
                               const std::vector<double>& affective_features,
                               const std::vector<double>& person_features);
};

/**
 * @brief Network for analyzing individual message elements
 * 
 * Architecture: Evaluates effectiveness of headlines, images, CTAs, etc.
 */
class ElementAnalysisNetwork : public NeuralNetwork {
public:
    ElementAnalysisNetwork();
    
    double analyze_element(const MessageElement& element);
    std::vector<double> extract_element_features(const MessageElement& element);
};

/**
 * @brief LSTM-based network for sequential campaign analysis
 * 
 * Architecture: Processes sequences of touchpoints to understand
 * customer journey dynamics
 */
class CampaignEffectivenessNetwork {
private:
    struct LSTMCell {
        std::vector<double> hidden_state;
        std::vector<double> cell_state;
        int hidden_size;
        
        // LSTM gates
        std::vector<std::vector<double>> W_f, W_i, W_c, W_o;  // Weights
        std::vector<double> b_f, b_i, b_c, b_o;  // Biases
        
        LSTMCell(int input_size, int hidden_size);
        void forward(const std::vector<double>& input);
        void reset();
    };
    
    std::vector<LSTMCell> lstm_cells;
    int hidden_size;
    int num_layers;

public:
    CampaignEffectivenessNetwork(int input_size, int hidden_size = 128, int num_layers = 2);
    
    std::vector<double> process_sequence(const std::vector<TouchPoint>& touchpoints);
    void reset();
};

// ============================================================================
// MODULE 4: BAYESIAN NETWORK
// ============================================================================
// This module implements probabilistic reasoning for handling uncertainty
// in marketing response prediction

/**
 * @brief Node in a Bayesian network representing a random variable
 */
class BayesianNode {
private:
    std::string name;
    std::vector<std::string> states;
    std::vector<std::string> parents;
    std::map<std::vector<std::string>, std::map<std::string, double>> cpt;  // Conditional Probability Table

public:
    BayesianNode() = default;  // Default constructor for map usage
    BayesianNode(const std::string& name, const std::vector<std::string>& states);
    
    void add_parent(const std::string& parent_name);
    void set_probability(const std::vector<std::string>& parent_states,
                        const std::string& state, double prob);
    
    double get_probability(const std::vector<std::string>& parent_states,
                          const std::string& state) const;
    
    const std::string& get_name() const { return name; }
    const std::vector<std::string>& get_states() const { return states; }
};

/**
 * @brief Bayesian network for probabilistic inference
 * 
 * Purpose: Models causal relationships and uncertainty in the
 * marketing response process
 */
class BayesianNetwork {
private:
    std::map<std::string, BayesianNode> nodes;
    std::map<std::string, std::string> evidence;

public:
    BayesianNetwork() = default;
    
    void add_node(const BayesianNode& node);
    void set_evidence(const std::string& node_name, const std::string& state);
    void clear_evidence();
    
    std::map<std::string, double> query(const std::string& node_name);
    void initialize_marketing_network();
};

// ============================================================================
// MODULE 5: MAIN AGENT
// ============================================================================
// This module integrates all components into a unified marketing response
// prediction and analysis system

/**
 * @brief Main integrated agent for marketing response prediction
 * 
 * This class combines all neural networks and the Bayesian network to provide:
 * - Response prediction for individual messages
 * - Campaign effectiveness analysis
 * - Element-level analysis
 * - Training and model persistence
 */
class EnhancedMarketingAgent {
private:
    // Neural network components
    std::unique_ptr<CognitiveProcessingNetwork> cognitive_network;
    std::unique_ptr<AffectiveProcessingNetwork> affective_network;
    std::unique_ptr<DecisionNetwork> decision_network;
    std::unique_ptr<ElementAnalysisNetwork> element_network;
    std::unique_ptr<CampaignEffectivenessNetwork> campaign_network;
    
    // Bayesian network for probabilistic reasoning
    BayesianNetwork bayesian_network;
    
    // Response categories
    enum class Response {
        IGNORE,      // Message not noticed or immediately dismissed
        CONSIDER,    // Message noticed, minimal engagement
        ENGAGE,      // Active engagement with message content
        CONVERT,     // Desired action taken
        ADVOCATE     // Positive sharing/advocacy
    };
    
    // Helper methods
    std::vector<double> extract_person_features(const Person& person);
    Response vector_to_response(const std::vector<double>& output);
    std::string response_to_string(Response resp);
    double calculate_temporal_decay(double time_offset, double decay_rate = 0.1);
    std::map<std::string, double> calculate_channel_synergy(
        const std::vector<TouchPoint>& touchpoints);

public:
    EnhancedMarketingAgent();
    
    /**
     * @brief Predict response to a marketing message
     * @param message The marketing message to analyze
     * @param person The target person's characteristics
     * @return Predicted response category as string
     */
    std::string predict_response(const Message& message, const Person& person);
    
    /**
     * @brief Analyze effectiveness of a multi-touchpoint campaign
     * @param touchpoints Sequence of marketing interactions
     * @param person Target person's characteristics
     * @return Detailed campaign analysis
     */
    CampaignAnalysis analyze_campaign(const std::vector<TouchPoint>& touchpoints,
                                     const Person& person);
    
    /**
     * @brief Analyze effectiveness of a specific message element
     * @param element Individual message component (headline, image, etc.)
     * @return Effectiveness score (0-1)
     */
    double analyze_element(const MessageElement& element);
    
    /**
     * @brief Train the agent on historical data
     * @param messages Training messages
     * @param people Associated people
     * @param responses Observed responses
     * @param epochs Number of training iterations
     * @param learning_rate Learning rate for gradient descent
     */
    void train(const std::vector<Message>& messages,
              const std::vector<Person>& people,
              const std::vector<std::string>& responses,
              int epochs = 100,
              double learning_rate = 0.001);
    
    /**
     * @brief Save trained models to disk
     * @param directory Directory path for model files
     */
    void save_models(const std::string& directory);
    
    /**
     * @brief Load trained models from disk
     * @param directory Directory path containing model files
     */
    void load_models(const std::string& directory);
};

} // namespace marketing

#endif // ENHANCED_MARKETING_AGENT_HPP

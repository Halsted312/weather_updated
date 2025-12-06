# Enhanced Marketing Agent - Module Navigation Guide

## Quick Module Reference

This guide helps developers quickly locate specific functionality within the C++ codebase.

---

## 📁 File Structure

```
enhanced_marketing_agent.hpp    - All class declarations with module markers
enhanced_marketing_agent.cpp    - All implementations with module markers
demo.cpp                        - Usage examples
CMakeLists.txt                  - CMake build configuration
Makefile                        - Make build configuration
```

---

## 🗺️ Module Index

### MODULE 1: Core Data Structures
**Location**: `enhanced_marketing_agent.hpp` lines 50-180

**Purpose**: Fundamental data types for marketing messages and people

**Key Classes**:
- `Message` - Marketing message with content and metadata
- `Person` - Target audience member with demographics
- `MessageElement` - Individual message component (headline, image, CTA)
- `CampaignAnalysis` - Results of campaign effectiveness analysis
- `TouchPoint` - Single interaction in customer journey

**When to use**: 
- Creating marketing messages
- Defining target audiences
- Building campaign sequences
- Analyzing message components

---

### MODULE 2: Neural Network Foundation
**Location**: 
- Header: `enhanced_marketing_agent.hpp` lines 180-320
- Implementation: `enhanced_marketing_agent.cpp` [MODULE 2 IMPL]

**Purpose**: Basic neural network building blocks

**Key Classes**:
- `Neuron` - Single neuron with activation function
- `Layer` - Layer of neurons with forward/backward pass
- `NeuralNetwork` - Multi-layer feedforward network

**Key Methods**:
- `forward()` - Forward propagation
- `backward()` - Backpropagation for training
- `train()` - Training with gradient descent

**When to use**:
- Building custom neural network architectures
- Understanding base network operations
- Implementing new specialized networks

---

### MODULE 3: Specialized Neural Networks
**Location**:
- Header: `enhanced_marketing_agent.hpp` lines 320-550
- Implementation: `enhanced_marketing_agent.cpp` [MODULE 3 IMPL]

**Purpose**: Domain-specific neural network architectures

**Key Classes**:

#### CognitiveProcessingNetwork
- **Purpose**: Analyzes message content for clarity and relevance
- **Input**: Message text
- **Output**: Cognitive feature vector (64 dims)
- **Key Method**: `process(message)`

#### AffectiveProcessingNetwork
- **Purpose**: Analyzes emotional impact
- **Input**: Message content
- **Output**: Emotional feature vector (32 dims)
- **Key Method**: `process(message)`

#### DecisionNetwork
- **Purpose**: Predicts behavioral response
- **Input**: Cognitive + Affective + Person features
- **Output**: Response probability distribution (5 categories)
- **Key Method**: `predict(cognitive, affective, person)`

#### ElementAnalysisNetwork
- **Purpose**: Evaluates individual message elements
- **Input**: MessageElement (headline, image, etc.)
- **Output**: Effectiveness score (0-1)
- **Key Method**: `analyze_element(element)`

#### CampaignEffectivenessNetwork
- **Purpose**: Analyzes sequential customer journeys (LSTM-based)
- **Input**: Sequence of TouchPoints
- **Output**: Campaign feature vector
- **Key Method**: `process_sequence(touchpoints)`

**When to use**:
- Processing marketing messages
- Analyzing emotional impact
- Predicting customer responses
- Evaluating campaign sequences

---

### MODULE 4: Bayesian Network
**Location**:
- Header: `enhanced_marketing_agent.hpp` lines 550-650
- Implementation: `enhanced_marketing_agent.cpp` [MODULE 4 IMPL]

**Purpose**: Probabilistic reasoning and uncertainty modeling

**Key Classes**:
- `BayesianNode` - Single node with conditional probability table
- `BayesianNetwork` - Complete Bayesian network with inference

**Key Methods**:
- `add_node()` - Add a probabilistic variable
- `set_evidence()` - Set observed values
- `query()` - Perform probabilistic inference
- `initialize_marketing_network()` - Build marketing domain network

**When to use**:
- Modeling causal relationships
- Handling uncertainty
- Understanding factor interactions
- Probabilistic predictions

---

### MODULE 5: Main Agent
**Location**:
- Header: `enhanced_marketing_agent.hpp` lines 650-end
- Implementation: `enhanced_marketing_agent.cpp` [MODULE 5 IMPL]

**Purpose**: Integrated marketing response prediction system

**Key Class**: `EnhancedMarketingAgent`

**Key Methods**:

#### predict_response()
```cpp
std::string predict_response(const Message& message, const Person& person)
```
- **Purpose**: Predict single message response
- **Returns**: "ignore", "consider", "engage", "convert", or "advocate"
- **Use case**: A/B testing, message optimization

#### analyze_campaign()
```cpp
CampaignAnalysis analyze_campaign(
    const std::vector<TouchPoint>& touchpoints,
    const Person& person
)
```
- **Purpose**: Analyze multi-touchpoint campaign effectiveness
- **Returns**: Comprehensive campaign analysis with metrics
- **Use case**: Campaign evaluation, optimization recommendations

#### analyze_element()
```cpp
double analyze_element(const MessageElement& element)
```
- **Purpose**: Evaluate individual message component
- **Returns**: Effectiveness score (0-1)
- **Use case**: Creative testing, element optimization

#### train()
```cpp
void train(
    const std::vector<Message>& messages,
    const std::vector<Person>& people,
    const std::vector<std::string>& responses,
    int epochs = 100,
    double learning_rate = 0.001
)
```
- **Purpose**: Train agent on historical data
- **Use case**: Model customization, domain adaptation

#### save_models() / load_models()
```cpp
void save_models(const std::string& directory)
void load_models(const std::string& directory)
```
- **Purpose**: Persist trained models
- **Use case**: Deployment, model versioning

**When to use**:
- Main interface for all predictions
- Campaign analysis and optimization
- Training and deployment

---

## 🔍 Finding Specific Functionality

### "I want to predict how someone will respond to a message"
→ **MODULE 5**: `EnhancedMarketingAgent::predict_response()`

### "I want to analyze a multi-step campaign"
→ **MODULE 5**: `EnhancedMarketingAgent::analyze_campaign()`

### "I want to test different headlines"
→ **MODULE 5**: `EnhancedMarketingAgent::analyze_element()`

### "I want to understand emotional impact"
→ **MODULE 3**: `AffectiveProcessingNetwork::process()`

### "I want to add custom features to messages"
→ **MODULE 1**: `Message` struct, modify `features` map

### "I want to build a custom neural network"
→ **MODULE 2**: Inherit from `NeuralNetwork` base class

### "I want to add probabilistic reasoning"
→ **MODULE 4**: `BayesianNetwork` class

### "I want to process sequential data"
→ **MODULE 3**: `CampaignEffectivenessNetwork` (LSTM)

---

## 📊 Response Categories

The agent predicts one of five response categories:

1. **IGNORE** - Message not noticed or dismissed immediately
2. **CONSIDER** - Message noticed with minimal engagement
3. **ENGAGE** - Active engagement with content
4. **CONVERT** - Desired action taken (purchase, signup, etc.)
5. **ADVOCATE** - Positive sharing/advocacy behavior

---

## 🛠️ Common Development Patterns

### Adding a New Feature to Messages
1. Go to **MODULE 1** (`Message` struct)
2. Add to `features` map or create new member variable
3. Update feature extraction in **MODULE 5** (`extract_person_features()`)

### Creating a Custom Neural Network
1. Go to **MODULE 2** 
2. Inherit from `NeuralNetwork` base class
3. Implement in **MODULE 3** section
4. Add architecture in constructor
5. Implement domain-specific `process()` method

### Extending the Agent
1. Go to **MODULE 5** (`EnhancedMarketingAgent`)
2. Add new private member for your component
3. Initialize in constructor
4. Add public method for new functionality

---

## 🚀 Quick Start Code Locations

### Example 1: Basic Prediction
See `demo.cpp` - Simple prediction example

### Example 2: Campaign Analysis
See `demo.cpp` - Multi-touchpoint campaign

### Example 3: Element Testing
See `demo.cpp` - A/B testing headlines

### Example 4: Training
See **MODULE 5**: `train()` method implementation

---

## 📝 Code Style Guide

### Searching in the Code

Use these markers to jump to sections:
```
[MODULE 1 IMPL]  - Core structures implementation
[MODULE 2 IMPL]  - Neural network foundation
[MODULE 3 IMPL]  - Specialized networks
[MODULE 4 IMPL]  - Bayesian network
[MODULE 5 IMPL]  - Main agent
```

### In Your IDE

1. **VSCode**: Use `Ctrl+Shift+F` to search for module markers
2. **CLion**: Use `Ctrl+Shift+F` or Structure view
3. **Vim**: Use `/MODULE X IMPL` to jump to sections

---

## 🔄 Workflow Examples

### Workflow 1: Optimize a Marketing Message
1. Create `Message` (**MODULE 1**)
2. Create target `Person` (**MODULE 1**)
3. Call `predict_response()` (**MODULE 5**)
4. Iterate message content
5. Test with `analyze_element()` for components (**MODULE 5**)

### Workflow 2: Analyze Campaign Performance
1. Create multiple `TouchPoint` objects (**MODULE 1**)
2. Define sequence with time offsets
3. Call `analyze_campaign()` (**MODULE 5**)
4. Review `CampaignAnalysis` results
5. Implement recommendations

### Workflow 3: Train on Custom Data
1. Prepare training data (messages, people, responses)
2. Create `EnhancedMarketingAgent` (**MODULE 5**)
3. Call `train()` with your data (**MODULE 5**)
4. Call `save_models()` to persist (**MODULE 5**)
5. Deploy with `load_models()` (**MODULE 5**)

---

## 📚 Additional Resources

- **Build Instructions**: See `README.md`
- **API Examples**: See `demo.cpp`
- **Architecture Diagram**: See `README.md`

---

## 🎯 Performance Tips

### For Fast Inference
- Use **MODULE 5** methods directly
- Pre-extract person features once
- Batch process multiple messages

### For Training
- Start with **MODULE 2** to understand base operations
- Implement custom loss functions in **MODULE 3**
- Use **MODULE 5** `train()` as template

### For Analysis
- Use **MODULE 4** Bayesian network for uncertainty
- Combine **MODULE 3** networks for multi-aspect analysis
- Use **MODULE 5** campaign analysis for sequences

---

*Last Updated: 2024*
*For questions or contributions, see project documentation*

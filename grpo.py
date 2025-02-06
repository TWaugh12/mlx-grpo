import mlx.core as mx

class GRPO:
    def __init__(self, model, tokenizer, reward_funcs, config):
        """
        Args:
            model: LLM
            tokenizer: Tokenizer with .encode and .decode methods.
            reward_funcs: A list of reward functions.
            config: Configuration object with attributes:
                - num_generations: Number of completions per prompt.
                - generation_temperature: Sampling temperature.
                - beta: KL penalty coefficient.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.reward_funcs = reward_funcs  
        self.config = config

    def generate_completions(self, prompts):
        completions = []
        for prompt in prompts:
            input_tokens = self.tokenizer.encode(prompt, return_tensors="np").tolist()
            sample_completions = []
            for _ in range(self.config.num_generations):
                tokens = mx.array(input_tokens)
                tokens_gen = []
                for token in self.model.generate(tokens, temp=self.config.generation_temperature):
                    mx.eval(token)
                    if token.item() == self.tokenizer.eos_token_id:
                        break
                    tokens_gen.append(token.item())
                output_text = self.tokenizer.decode(tokens_gen)
                sample_completions.append(output_text)
            completions.append(sample_completions)
        return completions

    def compute_rewards(self, prompts, completions, targets, additional_info_batch):
        """
        Computes rewards for each completion from each prompt.
        Each reward function is applied over flattened lists of completions,
        prompts, targets, and additional info. Rewards from multiple reward
        functions are then averaged.
        """
        rewards_per_prompt = [[] for _ in range(len(prompts))]

        for reward_func in self.reward_funcs:
            # Flatten lists: each prompt contributes self.config.num_generations items.
            flattened_completions = [comp for sublist in completions for comp in sublist]
            flattened_prompts = [prompt for prompt in prompts for _ in range(self.config.num_generations)]
            flattened_targets = [target for target in targets for _ in range(self.config.num_generations)]
            flattened_info = [info for info in additional_info_batch for _ in range(self.config.num_generations)]

            rewards = reward_func(
                completions=flattened_completions,
                targets=flattened_targets,
                prompts=flattened_prompts,
                additional_info=flattened_info
            )
            for i in range(len(prompts)):
                start_index = i * self.config.num_generations
                end_index = (i + 1) * self.config.num_generations
                rewards_per_prompt[i].append(rewards[start_index:end_index])

        averaged_rewards = []
        for prompt_rewards in rewards_per_prompt:
            gen_rewards = []
            for gen_index in range(self.config.num_generations):
                r_list = [func_rewards[gen_index] for func_rewards in prompt_rewards]
                avg_r = sum(r_list) / len(r_list) if r_list else 0.0
                gen_rewards.append(avg_r)
            averaged_rewards.append(gen_rewards)
        return averaged_rewards

    def compute_kl_divergence(self, model_outputs, old_model_outputs):
        """
        Computes the KL divergence between the current policy and the old policy.
        Each element is a 1D array of logits. 
        The divergence is computed for each prompt and generation.
        """
        kl_divergences = []
        for current_outputs, old_outputs in zip(model_outputs, old_model_outputs):
            prompt_kls = []
            for current_logits, old_logits in zip(current_outputs, old_outputs):
                probs = mx.softmax(current_logits, axis=-1)
                old_probs = mx.softmax(old_logits, axis=-1)
                kl = mx.sum(old_probs * mx.log(old_probs / probs), axis=-1).mean()
                prompt_kls.append(kl)
            kl_divergences.append(prompt_kls)
        return kl_divergences

    def compute_advantages_and_loss(self, rewards, model_outputs, old_model_outputs):
        """
        Computes normalized advantages per prompt and GRPO loss as:

            loss = - advantage + beta * KL

        A single gradient step is used so that the probability ratio is 1.
        """
        advantages = []  # One list per prompt (length == num_generations)
        losses = []      # One loss value per completion

        # Compute normalized advantages per prompt.
        for prompt_rewards in rewards:
            mean_r = sum(prompt_rewards) / len(prompt_rewards)
            std_r = (sum((r - mean_r) ** 2 for r in prompt_rewards) / len(prompt_rewards)) ** 0.5 + 1e-8
            prompt_advantages = [(r - mean_r) / std_r for r in prompt_rewards]
            advantages.append(prompt_advantages)

        # Compute KL divergence penalties.
        kl_penalties = self.compute_kl_divergence(model_outputs, old_model_outputs)

        # Form the per-completion loss.
        for prompt_advantages, prompt_kls in zip(advantages, kl_penalties):
            for adv, kl in zip(prompt_advantages, prompt_kls):
                losses.append(-adv + self.config.beta * kl)
        return mx.array(losses).mean() if losses else mx.array(0.0)

    def get_model_outputs(self, prompts):
        """
        Computes the logits of the last token for each generation.
        """
        all_outputs = []
        for prompt in prompts:
            input_tokens = self.tokenizer.encode(prompt, return_tensors="np").tolist()
            prompt_outputs = []
            for _ in range(self.config.num_generations):
                tokens = mx.array(input_tokens)
                logits = self.model(tokens[None])[0, -1, :]
                prompt_outputs.append(logits)
            all_outputs.append(prompt_outputs)
        return all_outputs

    def compute_loss(self, prompts, targets, additional_info_batch):
        completions = self.generate_completions(prompts)

        with mx.no_grad():
            old_model_outputs = self.get_model_outputs(prompts)
        model_outputs = self.get_model_outputs(prompts)

        rewards = self.compute_rewards(prompts, completions, targets, additional_info_batch)
        loss_value = self.compute_advantages_and_loss(rewards, model_outputs, old_model_outputs)
        return loss_value
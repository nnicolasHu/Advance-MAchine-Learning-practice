from textloader import  string2code, id2lettre
import math
import torch

#  TODO:  Ce fichier contient les différentes fonction de génération

def generate(rnn, emb, decoder, eos, start="", maxlen=200):
    """  Fonction de génération (l'embedding et le decodeur être des fonctions du rnn). Initialise le réseau avec start (ou à 0 si start est vide) et génère une séquence de longueur maximale 200 ou qui s'arrête quand eos est généré.
        * rnn : le réseau
        * emb : la couche d'embedding
        * decoder : le décodeur
        * eos : ID du token end of sequence
        * start : début de la phrase
        * maxlen : longueur maximale
    """

    #  TODO:  Implémentez la génération à partir du RNN, et d'une fonction decoder qui renvoie les logits
    # (logarithme de probabilité à une constante près, i.e. ce qui vient avant le softmax) des différentes sorties possibles
    rnn.eval()
    with torch.no_grad():
        if start:
            input = string2code(start)
            hidden = None  
        else:
            input = torch.zeros(1, dtype=torch.long)   
            hidden = None
        
        generated_sequence = []

        for _ in range(maxlen):
            embedded_input = emb(input)
            output, hidden = rnn(embedded_input, hidden)

            logits = decoder(output)
            probabilities = F.softmax(logits, dim=-1)

            # For deterministic generation:
            # next_char = probabilities.argmax(dim=-1)
            # For random generationi:
            next_char = torch.multinomial(probabilities, num_samples=1)

            generated_sequence.append(next_char.item())

            if next_char.item() == eos:
                break

            input = next_char

        return ''.join(id2lettre[c] for c in generated_sequence)



def generate_beam(rnn, emb, decoder, eos, k, start="", maxlen=200):
    """
        Génere une séquence en beam-search : à chaque itération, on explore pour chaque candidat les k symboles les plus probables; puis seuls les k meilleurs candidats de l'ensemble des séquences générées sont conservés (au sens de la vraisemblance) pour l'itération suivante.
        * rnn : le réseau
        * emb : la couche d'embedding
        * decoder : le décodeur
        * eos : ID du token end of sequence
        * k : le paramètre du beam-search
        * start : début de la phrase
        * maxlen : longueur maximale
    """
    #  TODO:  Implémentez le beam Search


# p_nucleus
def p_nucleus(decoder, alpha: float):
    """Renvoie une fonction qui calcule la distribution de probabilité sur les sorties

    Args:
        * decoder: renvoie les logits étant donné l'état du RNN
        * alpha (float): masse de probabilité à couvrir
    """
    def compute(h):
        """Calcule la distribution de probabilité sur les sorties

        Args:
           * h (torch.Tensor): L'état à décoder
        """
        #  TODO:  Implémentez le Nucleus sampling ici (pour un état s)
    return compute

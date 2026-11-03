# Fixed application messages after a supported input language has been identified.

LANGUAGE_NAMES = {"en": "English", "pt": "Portuguese", "af": "Afrikaans"}

MESSAGES = {
    "en": {
        "invalid_input": "Enter a clear factual statement to check. Questions, instructions, opinions and meaningless text cannot be checked as claims.",
        "out_of_scope": "Enter a factual claim in English, Portuguese or Afrikaans. Greetings, help requests, creative tasks and other languages are outside this scope.",
        "no_evidence": "No evidence was retrieved. This claim could not be evaluated.",
        "no_evidence_reason": "No passages were retrieved; no model prediction was generated.",
        "encoding": "The English claim could not fit the model input. Use a shorter factual claim and retry. No text was truncated.",
        "service_error": "Language screening, translation or explanation could not be completed. Please retry.",
        "configuration": "Configure an explanation model and API key in Settings.",
        "authentication": "The provider rejected the API key. Update it in Settings.",
        "access": "The configured model is unavailable for this API key. Check model access in Settings.",
        "quota": "The provider's rate limit or API quota was reached. Check API usage and billing, then retry.",
        "provider": "The explanation provider is temporarily unavailable. Please retry later.",
        "timeout": "The language or explanation request timed out. Please retry the claim.",
        "connection": "The server could not connect to the provider. Check the internet connection.",
        "request": "The provider rejected the request. Choose a model that supports structured output in Settings.",
        "invalid_output": "The provider did not return a complete, valid translation and explanation. Retry or use fewer evidence passages.",
        "refusal": "The provider declined the language or explanation request. No prediction is displayed.",
        "notice": "BERT prediction based on retrieved Wikipedia passages. Model confidence is not a probability that a claim is true. Evidence sufficiency has not yet been calibrated for this application.",
    },
    "pt": {
        "invalid_input": "Introduza uma afirmação factual clara para verificar. Perguntas, instruções, opiniões e texto sem sentido não podem ser verificados como afirmações.",
        "out_of_scope": "Introduza uma afirmação factual em inglês, português ou africâner. Saudações, pedidos de ajuda, tarefas criativas e outros idiomas estão fora deste âmbito.",
        "no_evidence": "Não foram encontradas evidências. Não foi possível avaliar esta afirmação.",
        "no_evidence_reason": "Não foram encontradas passagens; não foi gerada uma previsão do modelo.",
        "encoding": "A afirmação em inglês excede o limite de entrada do modelo. Use uma afirmação factual mais curta e tente novamente. Nenhum texto foi truncado.",
        "service_error": "Não foi possível concluir a análise do idioma, a tradução ou a explicação. Tente novamente.",
        "configuration": "Configure um modelo de explicação e uma chave de API nas Definições (Settings).",
        "authentication": "O fornecedor rejeitou a chave de API. Atualize-a nas Definições (Settings).",
        "access": "O modelo configurado não está disponível para esta chave de API. Verifique o acesso nas Definições (Settings).",
        "quota": "Foi atingido o limite de pedidos ou a quota da API. Verifique a utilização e a faturação e tente novamente.",
        "provider": "O fornecedor de explicações está temporariamente indisponível. Tente novamente mais tarde.",
        "timeout": "O pedido de tradução ou explicação excedeu o tempo limite. Tente verificar a afirmação novamente.",
        "connection": "O servidor não conseguiu ligar-se ao fornecedor. Verifique a ligação à Internet.",
        "request": "O fornecedor rejeitou o pedido. Escolha nas Definições (Settings) um modelo que suporte saída estruturada.",
        "invalid_output": "O fornecedor não devolveu uma tradução e explicação completas e válidas. Tente novamente ou use menos passagens de evidência.",
        "refusal": "O fornecedor recusou o pedido de tradução ou explicação. Não é apresentada uma previsão.",
        "notice": "Previsão do BERT baseada em passagens da Wikipédia. A confiança do modelo não é a probabilidade de a afirmação ser verdadeira. A suficiência das evidências ainda não foi calibrada nesta aplicação.",
    },
    "af": {
        "invalid_input": "Voer 'n duidelike feitelike bewering in om na te gaan. Vrae, instruksies, menings en betekenislose teks kan nie as bewerings nagegaan word nie.",
        "out_of_scope": "Voer 'n feitelike bewering in Engels, Portugees of Afrikaans in. Groete, hulpversoeke, kreatiewe take en ander tale val buite hierdie omvang.",
        "no_evidence": "Geen bewyse is gevind nie. Hierdie bewering kon nie beoordeel word nie.",
        "no_evidence_reason": "Geen teksgedeeltes is gevind nie; geen modelvoorspelling is gemaak nie.",
        "encoding": "Die Engelse bewering oorskry die model se invoerlimiet. Gebruik 'n korter feitelike bewering en probeer weer. Geen teks is afgekap nie.",
        "service_error": "Die taalontleding, vertaling of verduideliking kon nie voltooi word nie. Probeer asseblief weer.",
        "configuration": "Stel 'n verduidelikingsmodel en API-sleutel in Instellings (Settings) op.",
        "authentication": "Die verskaffer het die API-sleutel verwerp. Werk dit in Instellings (Settings) by.",
        "access": "Die ingestelde model is nie vir hierdie API-sleutel beskikbaar nie. Kontroleer toegang in Instellings (Settings).",
        "quota": "Die verskaffer se versoeklimiet of API-kwota is bereik. Kontroleer gebruik en fakturering en probeer weer.",
        "provider": "Die verduidelikingsverskaffer is tydelik nie beskikbaar nie. Probeer later weer.",
        "timeout": "Die vertaal- of verduidelikingsversoek het uitgetel. Probeer die bewering weer.",
        "connection": "Die bediener kon nie met die verskaffer verbind nie. Kontroleer die internetverbinding.",
        "request": "Die verskaffer het die versoek verwerp. Kies in Instellings (Settings) 'n model wat gestruktureerde uitvoer ondersteun.",
        "invalid_output": "Die verskaffer het nie 'n volledige, geldige vertaling en verduideliking gelewer nie. Probeer weer of gebruik minder bewysteksgedeeltes.",
        "refusal": "Die verskaffer het die vertaal- of verduidelikingsversoek geweier. Geen voorspelling word vertoon nie.",
        "notice": "BERT-voorspelling op grond van teksgedeeltes uit Wikipedia. Modelvertroue is nie die waarskynlikheid dat 'n bewering waar is nie. Die toereikendheid van bewyse is nog nie vir hierdie toepassing gekalibreer nie.",
    },
}


def message(language, key):
    return MESSAGES.get(language, MESSAGES["en"]).get(key, MESSAGES.get(language, MESSAGES["en"])["service_error"])

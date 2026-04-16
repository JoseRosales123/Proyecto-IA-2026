const predictBtn = document.getElementById("predictBtn");
const clearBtn = document.getElementById("clearBtn");
const ticketId = document.getElementById("ticketId");
const ticketSubject = document.getElementById("ticketSubject");
const ticketText = document.getElementById("ticketText");
const statusMessage = document.getElementById("statusMessage");
const resultBox = document.getElementById("resultBox");
const predictedCategory = document.getElementById("predictedCategory");
const predictedLabel = document.getElementById("predictedLabel");
const categoryBadge = document.getElementById("categoryBadge");
const resultTicketId = document.getElementById("resultTicketId");
const resultSubject = document.getElementById("resultSubject");
const tokensResult = document.getElementById("tokensResult");
const scoresResult = document.getElementById("scoresResult");
const confidenceList = document.getElementById("confidenceList");
const exampleButtons = document.querySelectorAll(".example-btn");

const categoryLabels = {
    ACCOUNT: "Problema de cuenta",
    CANCEL: "Cancelación",
    CONTACT: "Contacto / seguimiento",
    DELIVERY: "Entrega",
    FEEDBACK: "Retroalimentación",
    INVOICE: "Factura",
    ORDER: "Pedido",
    PAYMENT: "Pago",
    REFUND: "Reembolso",
    SHIPPING: "Envío",
    SUBSCRIPTION: "Suscripción"
};

function generateTicketId() {
    const randomPart = Math.floor(1000 + Math.random() * 9000);
    const date = new Date();
    const datePart = [
        date.getFullYear(),
        String(date.getMonth() + 1).padStart(2, "0"),
        String(date.getDate()).padStart(2, "0")
    ].join("");
    return `TCK-${datePart}-${randomPart}`;
}

function setNewTicketId() {
    ticketId.value = generateTicketId();
}

function showStatus(message, type) {
    statusMessage.textContent = message;
    statusMessage.classList.remove("hidden", "error", "success");
    statusMessage.classList.add(type);
}

function hideStatus() {
    statusMessage.classList.add("hidden");
}

function resetResult() {
    resultBox.classList.add("hidden");
    predictedCategory.textContent = "-";
    predictedLabel.textContent = "-";
    categoryBadge.textContent = "-";
    resultTicketId.textContent = "-";
    resultSubject.textContent = "-";
    tokensResult.textContent = "";
    scoresResult.textContent = "";
    confidenceList.innerHTML = "";
}

function clearForm() {
    ticketSubject.value = "";
    ticketText.value = "";
    resetResult();
    hideStatus();
    setNewTicketId();
    ticketSubject.focus();
}

function buildRequestText() {
    const subject = ticketSubject.value.trim();
    const description = ticketText.value.trim();

    if (!subject) {
        return description;
    }

    return `${subject}. ${description}`.trim();
}

function normalizeLogScores(logScores) {
    const entries = Object.entries(logScores || {});
    if (!entries.length) {
        return [];
    }

    const maxScore = Math.max(...entries.map(([, value]) => value));
    const shifted = entries.map(([label, value]) => [label, Math.exp(value - maxScore)]);
    const total = shifted.reduce((sum, [, value]) => sum + value, 0);

    return shifted
        .map(([label, value]) => ({
            label,
            probability: total > 0 ? (value / total) * 100 : 0
        }))
        .sort((a, b) => b.probability - a.probability);
}

function renderConfidence(logScores) {
    const ranked = normalizeLogScores(logScores).slice(0, 5);
    confidenceList.innerHTML = "";

    ranked.forEach((item) => {
        const displayName = categoryLabels[item.label] || item.label;
        const row = document.createElement("div");
        row.className = "confidence-item";

        row.innerHTML = `
            <div class="confidence-name">${displayName}</div>
            <div class="bar-track">
                <div class="bar-fill" style="width: ${item.probability.toFixed(2)}%"></div>
            </div>
            <div class="confidence-value">${item.probability.toFixed(2)}%</div>
        `;

        confidenceList.appendChild(row);
    });
}

async function classifyTicket() {
    const description = ticketText.value.trim();
    const subject = ticketSubject.value.trim();
    const text = buildRequestText();

    resetResult();
    hideStatus();

    if (!description) {
        showStatus("Escribe una descripción antes de clasificar.", "error");
        return;
    }

    try {
        predictBtn.disabled = true;
        predictBtn.textContent = "Clasificando...";

        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text })
        });

        const data = await response.json();

        if (!response.ok) {
            showStatus(data.error || "Ocurrió un error al clasificar.", "error");
            return;
        }

        const code = data.predicted_category || "-";
        const label = categoryLabels[code] || code;

        predictedCategory.textContent = code;
        predictedLabel.textContent = label;
        categoryBadge.textContent = code;
        resultTicketId.textContent = ticketId.value;
        resultSubject.textContent = subject || "Sin asunto";
        tokensResult.textContent = Array.isArray(data.tokens) && data.tokens.length
            ? data.tokens.join(", ")
            : "Sin tokens";
        scoresResult.textContent = JSON.stringify(data.log_scores, null, 2);

        renderConfidence(data.log_scores);
        resultBox.classList.remove("hidden");
        showStatus("Clasificación realizada correctamente.", "success");
    } catch (error) {
        showStatus("No se pudo conectar con el backend.", "error");
    } finally {
        predictBtn.disabled = false;
        predictBtn.textContent = "Clasificar solicitud";
    }
}

predictBtn.addEventListener("click", classifyTicket);
clearBtn.addEventListener("click", clearForm);

ticketText.addEventListener("keydown", (event) => {
    if (event.ctrlKey && event.key === "Enter") {
        classifyTicket();
    }
});

exampleButtons.forEach((button) => {
    button.addEventListener("click", () => {
        ticketSubject.value = button.dataset.subject || "";
        ticketText.value = button.dataset.description || "";
        hideStatus();
        resetResult();
    });
});

setNewTicketId();